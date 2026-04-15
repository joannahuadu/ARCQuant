import torch
from torch import nn
from typing import List, Optional, Tuple
import math
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaAttention, LlamaMLP
from qLinearLayer import QLinearLayer
from quantize import *
from visualize import *
from x_mask import XMaskSwitchTop2Hard
from low_rank import LowRankResidual
import os
from optional_agemm import HAS_AGEMM, require_agemm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

@torch.no_grad()
def quantize_int_group(w, nbits, group_size):
    savedShape = w.shape
    w = w.reshape(-1, group_size)
    w_max = w.amax(dim=-1, keepdim=True)
    w_min = w.amin(dim=-1, keepdim=True)
    q_max = (2**(nbits)-1)
    q_min = (0)
    scales = (w_max-w_min).clamp(min=1e-5) / q_max
    base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
    w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    return w.reshape(savedShape)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def NVFP4_reorder_quantize_x(x, reorder_index, select_num):
    scale = torch.max(x.abs()).float() / (448.0*6.0)
    # scale = 1.0
    qx, scale_x = require_agemm().reorder_quantize_x(x / scale, reorder_index, select_num)
    return qx, scale_x, scale

def reorder_quantize_x(
    x,
    reorder_index,
    select_num,
    quant_type: str = "NVFP4",
    *,
    x_mask: Optional[nn.Module] = None,
    ste: bool = False,
    reorder_xw: bool = True,
    rec: bool = False,
):
    # If we need x_mask or gradients, fall back to the (differentiable via STE) fake path.
    if reorder_xw and quant_type == "NVFP4" and HAS_AGEMM and x_mask is None and not ste:
        return NVFP4_reorder_quantize_x(x, reorder_index, select_num)

    if reorder_xw:
        index = reorder_index.to(torch.int32)
        x_reordered = torch.index_select(x, 1, index)
        fake_reorder_index = torch.arange(x.shape[-1], device=x.device)
    else:
        x_reordered = x
        fake_reorder_index = reorder_index.to(device=x.device, dtype=torch.long)

    if x_mask is not None:
        if select_num > 0:
            if rec:
                x_rec_index = fake_reorder_index[-select_num:]
                # Preserve the reconstruction channels before x-mask modifies them.
                x_mask._last_x_rec = x_reordered[:, x_rec_index].detach()
                x_mask._last_x_rec_index = x_rec_index.detach().clone()
            else:
                x_mask._last_x_rec = None
                x_mask._last_x_rec_index = None
        x_reordered = x_mask(x_reordered)
    x_rec = getattr(x_mask, "_last_x_rec", None) if x_mask is not None else None
    return fake_reorder_quantize_x(
        x_reordered,
        fake_reorder_index,
        select_num,
        dtype=quant_type,
        ste=ste,
        x_rec=x_rec,
    )

class QLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        originalLayer: LlamaDecoderLayer,
        kv_cache,
        select_nums,
        reorder_index,
        layer_idx,
        quant_type,
        *,
        reorder_xw: bool = True,
        use_x_mask: bool = False,
        x_mask_tau: float = 1.0,
        x_mask_alpha: float = 1.0,
        x_mask_r_thr: Optional[float] = None,
        output_low_rank_rank: int = 0,
        mlp_output_low_rank_rank: int = 0,
        rec: bool = False,
    ):
        super().__init__()
       
        self.hidden_size = originalLayer.hidden_size
        self.self_attn = QLlamaAttention(
            originalLayer.self_attn,
            kv_cache,
            select_nums=select_nums,
            reorder_index=reorder_index,
            i=layer_idx,
            quant_type=quant_type,
            reorder_xw=reorder_xw,
            use_x_mask=use_x_mask,
            x_mask_tau=x_mask_tau,
            x_mask_alpha=x_mask_alpha,
            x_mask_r_thr=x_mask_r_thr,
            output_low_rank_rank=output_low_rank_rank,
            rec=rec,
        )
        # self.self_attn = originalLayer.self_attn
        self.mlp = QLlamaMLP(
            originalLayer.mlp,
            select_nums=select_nums,
            reorder_index=reorder_index,
            i=layer_idx,
            quant_type=quant_type,
            reorder_xw=reorder_xw,
            use_x_mask=use_x_mask,
            x_mask_tau=x_mask_tau,
            x_mask_alpha=x_mask_alpha,
            x_mask_r_thr=x_mask_r_thr,
            output_low_rank_rank=mlp_output_low_rank_rank,
            rec=rec,
        )
        # self.mlp = originalLayer.mlp
        self.input_layernorm = QLlamaRMSNorm(
            originalLayer.input_layernorm, 
            
        )
        self.post_attention_layernorm = QLlamaRMSNorm(
            originalLayer.post_attention_layernorm, 
            
        )

    def to(self, *args, **kwargs):
        super(QLlamaDecoderLayer, self).to(*args, **kwargs)
        self.self_attn = self.self_attn.to(*args, **kwargs)
        self.input_layernorm = self.input_layernorm.to(*args, **kwargs)
        self.post_attention_layernorm = self.post_attention_layernorm.to(*args, **kwargs)
        self.mlp = self.mlp.to(*args, **kwargs)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
   
        
class QLlamaRMSNorm(nn.Module):
    def __init__(
        self,
        originalNorm: LlamaRMSNorm,
    ):
        super().__init__()
        self.originalNorm = originalNorm
    

       
    def forward(self, hidden_states):
        result = self.originalNorm(hidden_states)
            
#         if self.args.abits < 16:
#             result = self.act_quant(result)
        
        
        return result
    
    def to(self, *args, **kwargs):
        super(QLlamaRMSNorm, self).to(*args, **kwargs)
        self.originalNorm = self.originalNorm.to(*args, **kwargs)
       
        return self

class QLlamaAttention(nn.Module):

    def __init__(
        self, 
        originalAttn: LlamaAttention,
        kv_cache,
        select_nums,
        reorder_index,
        i,
        quant_type,
        *,
        reorder_xw: bool = True,
        use_x_mask: bool = False,
        x_mask_tau: float = 1.0,
        x_mask_alpha: float = 1.0,
        x_mask_r_thr: Optional[float] = None,
        output_low_rank_rank: int = 0,
        rec: bool = False,
    ):
        super().__init__()
        self.q_kv_cache = kv_cache
        self.config = originalAttn.config
        self.hidden_size = originalAttn.hidden_size
        self.num_heads = originalAttn.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = originalAttn.num_key_value_heads
        self.num_key_value_groups = originalAttn.num_key_value_groups
        self.max_position_embeddings = originalAttn.max_position_embeddings
        self.rope_theta = originalAttn.rope_theta
        self.layer_idx = i
        self.quant_type = quant_type
        self.reorder_xw = bool(reorder_xw)
        self.rec = bool(rec)
        self.x_mask_in = (
            XMaskSwitchTop2Hard(
                self.hidden_size, x_mask_tau=x_mask_tau, x_mask_alpha=x_mask_alpha, x_mask_r_thr=x_mask_r_thr
            )
            if use_x_mask
            else None
        )
        self.x_mask_out = (
            XMaskSwitchTop2Hard(
                self.hidden_size, x_mask_tau=x_mask_tau, x_mask_alpha=x_mask_alpha, x_mask_r_thr=x_mask_r_thr
            )
            if use_x_mask
            else None
        )
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        nameTemplate = 'layers.{}.{}.{}.{}'
        self.q_proj = QLinearLayer(
            originalAttn.q_proj,
            select_num=select_nums[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'q_proj', 'input')],
            quant_type=quant_type,
            reorder_xw=self.reorder_xw,
        )
        self.k_proj = QLinearLayer(
            originalAttn.k_proj,
            select_num=select_nums[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'k_proj', 'input')],
            quant_type=quant_type,
            reorder_xw=self.reorder_xw,
        )
        self.v_proj = QLinearLayer(
            originalAttn.v_proj,
            select_num=select_nums[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'v_proj', 'input')],
            quant_type=quant_type,
            reorder_xw=self.reorder_xw,
        )
        self.o_proj = QLinearLayer(
            originalAttn.o_proj,
            select_num=select_nums[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'self_attn', 'o_proj', 'input')],
            quant_type=quant_type,
            reorder_xw=self.reorder_xw,
        )
        self.rotary_emb = originalAttn.rotary_emb

        self.register_buffer("softmax_alpha", torch.ones(self.num_heads, dtype=torch.float32))
        # Per-channel output scale applied after o_proj (position B).
        # Compensates for the channel-wise bias introduced by x_mask_out.
        # Applied to the dense o_proj output — does NOT disturb the 2:4 sparse qx
        # that feeds into the matrix multiply.
        self.register_buffer("output_scale", torch.ones(self.hidden_size, dtype=torch.float32))
        self.output_low_rank = None
        self.output_low_rank_rank = 0
        self.configure_output_low_rank(output_low_rank_rank)

        self.attention_dropout=originalAttn.attention_dropout

    def configure_output_low_rank(self, rank: int) -> None:
        rank = int(rank)
        if rank <= 0:
            self.output_low_rank = None
            self.output_low_rank_rank = 0
            return
        if self.output_low_rank is not None and int(getattr(self.output_low_rank, "rank", -1)) == rank:
            self.output_low_rank_rank = rank
            return
        self.output_low_rank = LowRankResidual(self.hidden_size, rank)
        self.output_low_rank_rank = rank

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def to(self, *args, **kwargs):
        super(QLlamaAttention, self).to(*args, **kwargs)
        self.q_proj = self.q_proj.to(*args, **kwargs)
        self.k_proj = self.k_proj.to(*args, **kwargs)
        self.v_proj = self.v_proj.to(*args, **kwargs)
        self.o_proj = self.o_proj.to(*args, **kwargs)
        self.rotary_emb = self.rotary_emb.to(*args, **kwargs)
        if self.output_low_rank is not None:
            self.output_low_rank = self.output_low_rank.to(*args, **kwargs)
      
        return self

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        

        residual_input = hidden_states
        bsz, q_len, _ = hidden_states.size()
        
        hidden_states = hidden_states.reshape(bsz * q_len, -1).contiguous()
        if not torch.is_grad_enabled():
            hidden_states = hidden_states.detach()
        qx, scale_x, scale = reorder_quantize_x(
            hidden_states,
            self.q_reorder_index,
            self.q_proj.select_num,
            self.quant_type,
            x_mask=self.x_mask_in,
            ste=torch.is_grad_enabled(),
            reorder_xw=self.reorder_xw,
            rec=self.rec,
        )
        hidden_states = (qx, scale_x, scale, bsz, q_len)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        
        # Fake quantize the key_states.
        # Preserve the position embedding info by first quantize.
        if self.q_kv_cache:
            key_states = quantize_int_group(key_states, nbits=4, group_size=64)
        
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
         
        else:
            cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # [bsz, nh, t, hd]
        query_states = (query_states.float() * self.softmax_alpha.view(1, self.num_heads, 1, 1)).to(
            dtype=query_states.dtype
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
       
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            
        if self.q_kv_cache:
            value_states = quantize_int_group(value_states, nbits=4, group_size=64)
            
            
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        
        # Quantize the attention output
      
        attn_output = attn_output.reshape(bsz * q_len, -1).contiguous()
        if not torch.is_grad_enabled():
            attn_output = attn_output.detach()


        qx, scale_x, scale = reorder_quantize_x(
            attn_output,
            self.o_reorder_index,
            self.o_proj.select_num,
            self.quant_type,
            x_mask=self.x_mask_out,
            ste=torch.is_grad_enabled(),
            reorder_xw=self.reorder_xw,
            rec=self.rec,
        )
        attn_output = (qx, scale_x, scale, bsz, q_len)
        attn_output = self.o_proj(attn_output)  # [bsz, q_len, hidden_size], dense

        # Position B: per-channel scale on the dense o_proj output.
        # The 2:4 sparse matmul is already done; this is a plain dense correction.
        attn_output = (attn_output.float() * self.output_scale).to(attn_output.dtype)
        if self.output_low_rank is not None:
            attn_output = attn_output + self.output_low_rank(residual_input)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    

class QLlamaMLP(nn.Module):
    def __init__(
        self,
        originalMLP: LlamaMLP,
        select_nums,
        reorder_index,
        i,
        quant_type,
        *,
        reorder_xw: bool = True,
        use_x_mask: bool = False,
        x_mask_tau: float = 1.0,
        x_mask_alpha: float = 1.0,
        x_mask_r_thr: Optional[float] = None,
        output_low_rank_rank: int = 0,
        rec: bool = False,
    ):
        super().__init__()
        nameTemplate = 'layers.{}.{}.{}.{}'

        self.quant_type = quant_type
        self.reorder_xw = bool(reorder_xw)
        self.rec = bool(rec)
        self.x_mask_up = (
            XMaskSwitchTop2Hard(
                originalMLP.up_proj.in_features,
                x_mask_tau=x_mask_tau,
                x_mask_alpha=x_mask_alpha,
                x_mask_r_thr=x_mask_r_thr,
            )
            if use_x_mask
            else None
        )
        self.x_mask_down = (
            XMaskSwitchTop2Hard(
                originalMLP.down_proj.in_features,
                x_mask_tau=x_mask_tau,
                x_mask_alpha=x_mask_alpha,
                x_mask_r_thr=x_mask_r_thr,
            )
            if use_x_mask
            else None
        )
        
        self.gate_proj = QLinearLayer(
            originalMLP.gate_proj,
            select_num=select_nums[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'gate_proj', 'input')],
            out_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            quant_type=self.quant_type,
            reorder_xw=self.reorder_xw,
        )
        self.down_proj = QLinearLayer(
            originalMLP.down_proj,
            select_num=select_nums[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            quant_type=self.quant_type,
            reorder_xw=self.reorder_xw,
        )
        self.up_proj = QLinearLayer(
            originalMLP.up_proj,
            select_num=select_nums[nameTemplate.format(i, 'mlp', 'up_proj', 'input')],
            reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'up_proj', 'input')],
            out_reorder_index=reorder_index[nameTemplate.format(i, 'mlp', 'down_proj', 'input')],
            quant_type=self.quant_type,
            reorder_xw=self.reorder_xw,
        )
        # Per-channel output scale applied after the dense down_proj output.
        self.register_buffer("mlp_output_scale", torch.ones(originalMLP.down_proj.out_features, dtype=torch.float32))
        self.output_low_rank = None
        self.output_low_rank_rank = 0
        self.configure_output_low_rank(output_low_rank_rank)
        self.act_fn = originalMLP.act_fn
        self.layer_idx = i
        
        
    def to(self, *args, **kwargs):
        super(QLlamaMLP, self).to(*args, **kwargs)
        self.gate_proj = self.gate_proj.to(*args, **kwargs)
        self.down_proj = self.down_proj.to(*args, **kwargs)
        self.up_proj = self.up_proj.to(*args, **kwargs)
        if self.output_low_rank is not None:
            self.output_low_rank = self.output_low_rank.to(*args, **kwargs)
        

        return self

    def forward(self, x):
        # input X: [b, seq, dim]: quantized

        residual_input = x
        bsz, q_len, _ = x.shape
        x = x.reshape(bsz * q_len, -1).contiguous()
        if not torch.is_grad_enabled():
            x = x.detach()

        qx, scale_x, scale = reorder_quantize_x(
            x,
            self.up_reorder_index,
            self.up_proj.select_num,
            self.quant_type,
            x_mask=self.x_mask_up,
            ste=torch.is_grad_enabled(),
            reorder_xw=self.reorder_xw,
            rec=self.rec,
        )
        x = (qx, scale_x, scale, bsz, q_len)
        tmpResult = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # Quantize the activations and feed into down_proj

        bsz, q_len, _ = tmpResult.shape
        tmpResult = tmpResult.reshape(bsz * q_len, -1).contiguous()
        if not torch.is_grad_enabled():
            tmpResult = tmpResult.detach()
        

        qx, scale_x, scale = reorder_quantize_x(
            tmpResult,
            self.down_reorder_index,
            self.down_proj.select_num,
            self.quant_type,
            x_mask=self.x_mask_down,
            ste=torch.is_grad_enabled(),
            reorder_xw=self.reorder_xw,
            rec=self.rec,
        )
        tmpResult = (qx, scale_x, scale, bsz, q_len)
        out = self.down_proj(tmpResult)
        out = (out.float() * self.mlp_output_scale).to(out.dtype)
        if self.output_low_rank is not None:
            out = out + self.output_low_rank(residual_input)
        return out
    def configure_output_low_rank(self, rank: int) -> None:
        rank = int(rank)
        if rank <= 0:
            self.output_low_rank = None
            self.output_low_rank_rank = 0
            return
        hidden_size = self.down_proj.out_features
        if self.output_low_rank is not None and int(getattr(self.output_low_rank, "rank", -1)) == rank:
            self.output_low_rank_rank = rank
            return
        self.output_low_rank = LowRankResidual(hidden_size, rank)
        self.output_low_rank_rank = rank
