from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class TokenResidualMLP(nn.Module):
    """Tiny per-token-per-group residual for x_mask gate logits.

    Input: xg [..., G, 4]
    Output: delta [..., G] (float32)
    """

    def __init__(self, hidden: int = 0, chunk_size: int = 1024):
        super().__init__()
        self.hidden = int(hidden)
        self.chunk_size = int(chunk_size) if chunk_size is not None else 0

        in_dim = 8  # [x(4), abs(x)(4)]
        if self.hidden <= 0:
            self.weight = nn.Parameter(torch.zeros((in_dim,), dtype=torch.float32), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=True)
            nn.init.trunc_normal_(self.weight, std=0.02)
        else:
            self.fc1 = nn.Linear(in_dim, self.hidden, bias=True, dtype=torch.float32)
            self.fc2 = nn.Linear(self.hidden, 1, bias=True, dtype=torch.float32)
            nn.init.trunc_normal_(self.fc1.weight, std=0.02)
            nn.init.zeros_(self.fc1.bias)
            nn.init.trunc_normal_(self.fc2.weight, std=0.02)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, xg: torch.Tensor) -> torch.Tensor:
        if xg.numel() == 0:
            return xg.new_zeros(*xg.shape[:-1], dtype=torch.float32)

        xg_flat = xg.reshape(-1, xg.shape[-2], 4)
        n, g = xg_flat.shape[0], xg_flat.shape[1]
        chunk = self.chunk_size if self.chunk_size and self.chunk_size > 0 else n
        out = xg_flat.new_empty((n, g), dtype=torch.float32)

        if self.hidden <= 0:
            w = self.weight
            w_x = w[:4]
            w_abs = w[4:]
            b = self.bias
            for i in range(0, n, chunk):
                xc = xg_flat[i : i + chunk].float()
                out[i : i + chunk] = (xc * w_x).sum(dim=-1) + (xc.abs() * w_abs).sum(dim=-1) + b
        else:
            for i in range(0, n, chunk):
                xc = xg_flat[i : i + chunk].float()
                feat = torch.cat([xc, xc.abs()], dim=-1)
                h = self.fc1(feat)
                h = F.gelu(h)
                out[i : i + chunk] = self.fc2(h).squeeze(-1)

        return out.view(*xg.shape[:-1])


class XMaskSwitchTop2Hard(nn.Module):
    """FlatQuant-style `switch_top2_hard` x-mask with optional token-level gating.

    This module applies 2:4 sparsity within each group of 4 channels:
    - Build a hard top-2 mask by magnitude, with optional soft relaxation (STE) via `x_mask_tau`.
    - Learn a per-group (and optional per-token) gate `r` in [0, 1] to mix dense vs sparse:
        mixed = r * x + (1 - r) * (x * mask)

    Token-level gate learning follows FlatQuant's `x_mask_token_gate_enabled` + TokenResidualMLP.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        group_size: int = 4,
        topk: int = 2,
        x_mask_tau: float = 1.0,
        x_mask_alpha: float = 1.0,
        x_mask_r_thr: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        group_size = int(group_size)
        topk = int(topk)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}")
        if group_size <= 0 or hidden_dim % group_size != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by group_size ({group_size})")
        if not (0 < topk <= group_size):
            raise ValueError(f"topk must be in (0, group_size]; got {topk} vs group_size={group_size}")

        self.use_x_mask = True
        self.x_mask_mode = "switch_top2_hard"
        self.group_size = group_size
        self.topk = topk
        self.x_mask_tau = float(x_mask_tau)
        self.x_mask_alpha = float(x_mask_alpha)
        self.x_mask_r_thr = x_mask_r_thr

        num_groups = hidden_dim // group_size
        self.x_mask_gate_logits = nn.Parameter(torch.zeros((num_groups,), dtype=torch.float32), requires_grad=True)
        self.x_mask_gate_mean_requires_grad = False

        self.x_mask_token_gate_enabled = False
        self.x_mask_token_mlp: Optional[TokenResidualMLP] = None
        self.x_mask_token_mlp_hidden = 0
        self.x_mask_token_mlp_chunk_size = 1024
        self.x_mask_token_use_layer_scale = True
        self.x_mask_token_scale = nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False)

        self._last_x_mask_ent = None
        self._last_x_mask_l2 = None
        self._last_x_mask_gate_mean = None
        self._last_x_mask_gate_mean_grad = None
        self._last_x_mask_gate_entropy = None
        self._last_x_mask_gate_std = None
        self._last_x_mask_gate_frac_low = None
        self._last_x_mask_gate_frac_high = None
        self._last_x_mask_gate_tok_var = None
        self._last_x_mask_gate_delta_l2 = None
        self._comp_channel_index = None
        self._comp_x = None

    def _ensure_x_mask_token_mlp(self) -> TokenResidualMLP:
        if self.x_mask_token_mlp is not None:
            if hasattr(self.x_mask_token_mlp, "chunk_size"):
                self.x_mask_token_mlp.chunk_size = int(self.x_mask_token_mlp_chunk_size)
            return self.x_mask_token_mlp
        hidden = int(getattr(self, "x_mask_token_mlp_hidden", 0))
        chunk = int(getattr(self, "x_mask_token_mlp_chunk_size", 1024))
        self.x_mask_token_mlp = TokenResidualMLP(hidden=hidden, chunk_size=chunk)
        return self.x_mask_token_mlp

    def _compute_x_mask_token_delta(self, reshaped: torch.Tensor) -> Optional[torch.Tensor]:
        if not getattr(self, "x_mask_token_gate_enabled", False):
            return None
        mlp = getattr(self, "x_mask_token_mlp", None)
        if mlp is None:
            mlp = self._ensure_x_mask_token_mlp()
        delta = mlp(reshaped)
        self._last_x_mask_gate_delta_l2 = delta.pow(2).mean()
        return delta

    def _compute_x_mask_gate_r(self, logits: torch.Tensor, reshaped: torch.Tensor):
        delta = self._compute_x_mask_token_delta(reshaped)
        if delta is None:
            r_fp32 = torch.sigmoid(logits.float())
            return r_fp32.to(dtype=logits.dtype), r_fp32

        # Token-level residual: zero-mean per token, bounded by tanh, scaled by a learnable scalar.
        delta = delta - delta.mean(dim=-1, keepdim=True)
        delta = torch.tanh(delta)
        scale = getattr(self, "x_mask_token_scale", None)
        if scale is not None:
            delta = delta * scale.to(delta)
        logits_fp32 = logits.float() + delta
        r_fp32 = torch.sigmoid(logits_fp32)
        return r_fp32.to(dtype=logits.dtype), r_fp32

    def _update_x_mask_gate_stats(self, r_fp32: torch.Tensor) -> None:
        if r_fp32 is None or r_fp32.numel() == 0:
            self._last_x_mask_gate_mean = None
            self._last_x_mask_gate_mean_grad = None
            self._last_x_mask_gate_std = None
            self._last_x_mask_gate_frac_low = None
            self._last_x_mask_gate_frac_high = None
            self._last_x_mask_gate_tok_var = None
            self._last_x_mask_gate_entropy = None
            return

        keep_grad_mean = bool(getattr(self, "x_mask_gate_mean_requires_grad", False))
        self._last_x_mask_gate_mean_grad = r_fp32.mean() if keep_grad_mean else None

        with torch.no_grad():
            stats = r_fp32.detach().float().cpu()
            self._last_x_mask_gate_mean = stats.mean()
            self._last_x_mask_gate_std = stats.std(unbiased=False)
            self._last_x_mask_gate_frac_low = (stats < 0.05).float().mean()
            self._last_x_mask_gate_frac_high = (stats > 0.95).float().mean()
            # token-wise variance averaged over groups
            r_flat = stats.reshape(-1, stats.shape[-1])
            mean_g = r_flat.mean(dim=0)
            var_g = r_flat.pow(2).mean(dim=0) - mean_g.pow(2)
            self._last_x_mask_gate_tok_var = var_g.clamp_min(0.0).mean()
        #     # Bernoulli entropy
        #     eps = 1e-12
        #     ent = -(stats * (stats + eps).log() + (1.0 - stats) * (1.0 - stats + eps).log())
        #     self._last_x_mask_gate_entropy = ent.mean()

    def _apply_x_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        self._last_x_mask_ent = None
        self._last_x_mask_l2 = None
        self._last_x_mask_gate_mean = None
        self._last_x_mask_gate_mean_grad = None
        self._last_x_mask_gate_entropy = None
        self._last_x_mask_gate_std = None
        self._last_x_mask_gate_frac_low = None
        self._last_x_mask_gate_frac_high = None
        self._last_x_mask_gate_tok_var = None
        self._last_x_mask_gate_delta_l2 = None

        alpha = float(getattr(self, "x_mask_alpha", 1.0))
        if alpha <= 0.0:
            return tensor

        # [*, G, 4]
        reshaped = tensor.view(*tensor.shape[:-1], -1, self.group_size)
        scores = reshaped.abs()

        tau = float(getattr(self, "x_mask_tau", 1.0))
        if tau <= 0.0:
            gate_soft = None
        else:
            p = torch.softmax(scores / tau, dim=-1)
            gate_soft = float(self.topk) * p

        idx = scores.topk(self.topk, dim=-1).indices
        gate_hard = torch.zeros_like(reshaped)
        gate_hard.scatter_(-1, idx, 1.0)
        gate_raw = gate_hard if gate_soft is None else gate_hard - gate_soft.detach() + gate_soft

        # 2-hot penalty (encourage exactly topk ones per group)
        self._last_x_mask_l2 = (gate_raw.pow(2).sum(dim=-1) - float(self.topk)).pow(2).mean()

        x_sp = reshaped * gate_raw

        # Per-group base logits, broadcast to token dims: [*, G]
        base = self.x_mask_gate_logits.to(device=tensor.device, dtype=torch.float32)
        base = base.view(*([1] * (reshaped.dim() - 2)), -1).expand(*reshaped.shape[:-1])
        r, r_fp32 = self._compute_x_mask_gate_r(base, reshaped)

        if not getattr(self, "_eval_mode", False):
            self._update_x_mask_gate_stats(r_fp32)

        r = r.to(dtype=reshaped.dtype).unsqueeze(-1)  # [*, G, 1]
        mixed = r * reshaped + (1.0 - r) * x_sp

        # Optional hard switch at eval time: enforce true 2:4 sparsity.
        r_thr = self.x_mask_r_thr
        if getattr(self, "_eval_mode", False) and r_thr is not None:
            hard_sel = (r_fp32 < float(r_thr)).to(dtype=mixed.dtype).unsqueeze(-1)
            mixed = mixed * (1.0 - hard_sel + hard_sel * gate_raw)

        if alpha < 1.0:
            mixed = (1.0 - alpha) * reshaped + alpha * mixed
        return mixed.view_as(tensor)

    def to_eval_mode(self) -> None:
        self._eval_mode = True

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "use_x_mask", False):
            return tensor
        if "switch_top2_hard" not in str(getattr(self, "x_mask_mode", "")):
            return tensor
        if tensor.shape[-1] % self.group_size != 0:
            return tensor
        return self._apply_x_mask(tensor)
