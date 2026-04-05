from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from x_mask import XMaskSwitchTop2Hard


def _configure_x_mask_module(
    module: Optional[XMaskSwitchTop2Hard],
    *,
    token_gate_mode: str,
    token_mlp_hidden: int,
    token_mlp_chunk_size: int,
    token_use_layer_scale: bool,
) -> None:
    if module is None:
        return
    if token_gate_mode == "token_all":
        module.x_mask_token_gate_enabled = True
        module.x_mask_token_mlp_hidden = int(token_mlp_hidden)
        module.x_mask_token_mlp_chunk_size = int(token_mlp_chunk_size)
        module.x_mask_token_use_layer_scale = bool(token_use_layer_scale)


def _make_input_mask_hook(owner: nn.Module, mask_name: str):
    def _hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        if not getattr(owner, "joint_plus_enabled", True):
            return None
        mask = getattr(owner, mask_name, None)
        if mask is None or not inputs:
            return None
        return (mask(inputs[0]),) + tuple(inputs[1:])

    return _hook


def _make_q_proj_output_hook(attn: nn.Module):
    def _hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
        if not getattr(attn, "joint_plus_enabled", True):
            return output
        alpha = getattr(attn, "softmax_alpha", None)
        if alpha is None:
            return output
        scale = alpha.to(device=output.device, dtype=output.dtype).repeat_interleave(attn.head_dim)
        return output * scale.view(*([1] * (output.dim() - 1)), -1)

    return _hook


def _make_output_scale_hook(owner: nn.Module, attr_name: str):
    def _hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
        if not getattr(owner, "joint_plus_enabled", True):
            return output
        output_scale = getattr(owner, attr_name, None)
        if output_scale is None:
            return output
        scale = output_scale.to(device=output.device, dtype=output.dtype)
        return output * scale.view(*([1] * (output.dim() - 1)), -1)

    return _hook


def _attach_joint_plus_to_attention(
    attn: nn.Module,
    *,
    use_x_mask: bool,
    x_mask_tau: float,
    x_mask_alpha: float,
    x_mask_r_thr: Optional[float],
    token_gate_mode: str,
    token_mlp_hidden: int,
    token_mlp_chunk_size: int,
    token_use_layer_scale: bool,
) -> None:
    ref_device = attn.q_proj.weight.device
    attn.x_mask_in = (
        XMaskSwitchTop2Hard(
            attn.hidden_size,
            x_mask_tau=x_mask_tau,
            x_mask_alpha=x_mask_alpha,
            x_mask_r_thr=x_mask_r_thr,
        ).to(device=ref_device)
        if use_x_mask
        else None
    )
    attn.x_mask_out = (
        XMaskSwitchTop2Hard(
            attn.hidden_size,
            x_mask_tau=x_mask_tau,
            x_mask_alpha=x_mask_alpha,
            x_mask_r_thr=x_mask_r_thr,
        ).to(device=ref_device)
        if use_x_mask
        else None
    )
    for module in (attn.x_mask_in, attn.x_mask_out):
        _configure_x_mask_module(
            module,
            token_gate_mode=token_gate_mode,
            token_mlp_hidden=token_mlp_hidden,
            token_mlp_chunk_size=token_mlp_chunk_size,
            token_use_layer_scale=token_use_layer_scale,
        )

    attn.register_buffer("softmax_alpha", torch.ones(attn.num_heads, dtype=torch.float32, device=ref_device))
    attn.register_buffer("output_scale", torch.ones(attn.hidden_size, dtype=torch.float32, device=ref_device))
    attn.joint_plus_enabled = True
    attn._joint_plus_hook_handles = [
        attn.q_proj.register_forward_pre_hook(_make_input_mask_hook(attn, "x_mask_in")),
        attn.k_proj.register_forward_pre_hook(_make_input_mask_hook(attn, "x_mask_in")),
        attn.v_proj.register_forward_pre_hook(_make_input_mask_hook(attn, "x_mask_in")),
        attn.o_proj.register_forward_pre_hook(_make_input_mask_hook(attn, "x_mask_out")),
        attn.q_proj.register_forward_hook(_make_q_proj_output_hook(attn)),
        attn.o_proj.register_forward_hook(_make_output_scale_hook(attn, "output_scale")),
    ]


def _attach_joint_plus_to_mlp(
    mlp: nn.Module,
    *,
    use_x_mask: bool,
    x_mask_tau: float,
    x_mask_alpha: float,
    x_mask_r_thr: Optional[float],
    token_gate_mode: str,
    token_mlp_hidden: int,
    token_mlp_chunk_size: int,
    token_use_layer_scale: bool,
    use_mlp_output_scale: bool,
) -> None:
    ref_device = mlp.gate_proj.weight.device
    hidden_size = getattr(mlp, "hidden_size", mlp.gate_proj.in_features)
    intermediate_size = getattr(mlp, "intermediate_size", mlp.down_proj.in_features)
    mlp.x_mask_up = (
        XMaskSwitchTop2Hard(
            hidden_size,
            x_mask_tau=x_mask_tau,
            x_mask_alpha=x_mask_alpha,
            x_mask_r_thr=x_mask_r_thr,
        ).to(device=ref_device)
        if use_x_mask
        else None
    )
    mlp.x_mask_down = (
        XMaskSwitchTop2Hard(
            intermediate_size,
            x_mask_tau=x_mask_tau,
            x_mask_alpha=x_mask_alpha,
            x_mask_r_thr=x_mask_r_thr,
        ).to(device=ref_device)
        if use_x_mask
        else None
    )
    for module in (mlp.x_mask_up, mlp.x_mask_down):
        _configure_x_mask_module(
            module,
            token_gate_mode=token_gate_mode,
            token_mlp_hidden=token_mlp_hidden,
            token_mlp_chunk_size=token_mlp_chunk_size,
            token_use_layer_scale=token_use_layer_scale,
        )

    if use_mlp_output_scale:
        mlp.register_buffer("mlp_output_scale", torch.ones(mlp.down_proj.out_features, dtype=torch.float32, device=ref_device))
    mlp.joint_plus_enabled = True
    mlp._joint_plus_hook_handles = [
        mlp.gate_proj.register_forward_pre_hook(_make_input_mask_hook(mlp, "x_mask_up")),
        mlp.up_proj.register_forward_pre_hook(_make_input_mask_hook(mlp, "x_mask_up")),
        mlp.down_proj.register_forward_pre_hook(_make_input_mask_hook(mlp, "x_mask_down")),
    ]
    if use_mlp_output_scale:
        mlp._joint_plus_hook_handles.append(
            mlp.down_proj.register_forward_hook(_make_output_scale_hook(mlp, "mlp_output_scale"))
        )


def apply_bf16_joint_plus(
    model,
    *,
    use_x_mask: bool = True,
    x_mask_tau: float = 1.0,
    x_mask_alpha: float = 1.0,
    x_mask_r_thr: Optional[float] = None,
    token_gate_mode: str = "token_all",
    token_mlp_hidden: int = 0,
    token_mlp_chunk_size: int = 1024,
    token_use_layer_scale: bool = True,
    use_mlp_output_scale: bool = False,
):
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and not getattr(attn, "_joint_plus_applied", False):
            if getattr(attn.config, "pretraining_tp", 1) > 1:
                raise NotImplementedError(
                    "bf16 hook mode requires pretraining_tp == 1; hook registration does not intercept "
                    "the F.linear fast path used when pretraining_tp > 1"
                )
            _attach_joint_plus_to_attention(
                attn,
                use_x_mask=use_x_mask,
                x_mask_tau=x_mask_tau,
                x_mask_alpha=x_mask_alpha,
                x_mask_r_thr=x_mask_r_thr,
                token_gate_mode=token_gate_mode,
                token_mlp_hidden=token_mlp_hidden,
                token_mlp_chunk_size=token_mlp_chunk_size,
                token_use_layer_scale=token_use_layer_scale,
            )
            attn._joint_plus_applied = True

        mlp = getattr(layer, "mlp", None)
        if (
            mlp is not None
            and hasattr(mlp, "gate_proj")
            and hasattr(mlp, "up_proj")
            and hasattr(mlp, "down_proj")
            and not getattr(mlp, "_joint_plus_applied", False)
        ):
            if getattr(mlp.config, "pretraining_tp", 1) > 1:
                raise NotImplementedError(
                    "bf16 hook mode requires pretraining_tp == 1; hook registration does not intercept "
                    "the F.linear fast path used when pretraining_tp > 1"
                )
            _attach_joint_plus_to_mlp(
                mlp,
                use_x_mask=use_x_mask,
                x_mask_tau=x_mask_tau,
                x_mask_alpha=x_mask_alpha,
                x_mask_r_thr=x_mask_r_thr,
                token_gate_mode=token_gate_mode,
                token_mlp_hidden=token_mlp_hidden,
                token_mlp_chunk_size=token_mlp_chunk_size,
                token_use_layer_scale=token_use_layer_scale,
                use_mlp_output_scale=use_mlp_output_scale,
            )
            mlp._joint_plus_applied = True
    return model


def set_layer_joint_plus_enabled(layer, enabled: bool) -> None:
    attn = getattr(layer, "self_attn", None)
    if attn is not None and hasattr(attn, "joint_plus_enabled"):
        attn.joint_plus_enabled = bool(enabled)
    mlp = getattr(layer, "mlp", None)
    if mlp is not None and hasattr(mlp, "joint_plus_enabled"):
        mlp.joint_plus_enabled = bool(enabled)


def split_bf16_joint_params(
    layer,
    *,
    train_alpha: bool,
    train_mlp_output_scale: bool = False,
) -> Tuple[list[nn.Parameter], list[nn.Parameter]]:
    gate_params: list[nn.Parameter] = []
    alpha_params: list[nn.Parameter] = []

    for module in iter_layer_x_mask_modules(layer):
        if hasattr(module, "x_mask_gate_logits"):
            module.x_mask_gate_logits.requires_grad_(True)
            gate_params.append(module.x_mask_gate_logits)
        if getattr(module, "x_mask_token_gate_enabled", False):
            mlp = module._ensure_x_mask_token_mlp()
            gate_params.extend(list(mlp.parameters()))
            scale = getattr(module, "x_mask_token_scale", None)
            if scale is not None:
                gate_params.append(scale)

    if train_alpha:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "softmax_alpha"):
            attn.softmax_alpha.requires_grad_(True)
            alpha_params.append(attn.softmax_alpha)
        if attn is not None and hasattr(attn, "output_scale"):
            attn.output_scale.requires_grad_(True)
            alpha_params.append(attn.output_scale)
        mlp = getattr(layer, "mlp", None)
        if train_mlp_output_scale and mlp is not None and hasattr(mlp, "mlp_output_scale"):
            mlp.mlp_output_scale.requires_grad_(True)
            alpha_params.append(mlp.mlp_output_scale)

    return gate_params, alpha_params


def save_bf16_joint_checkpoint(
    model,
    path: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    include_mlp_output_scale: bool = False,
) -> None:
    ckpt_layers: Dict[int, Dict[str, torch.Tensor]] = {}
    alpha_by_layer = []
    output_scale_by_layer = []
    mlp_output_scale_by_layer = []

    for idx, layer in enumerate(model.model.layers):
        keep = {}
        for key, value in layer.state_dict().items():
            if "x_mask" in key or key.endswith("softmax_alpha") or key.endswith("output_scale"):
                if not include_mlp_output_scale and key.endswith("mlp_output_scale"):
                    continue
                keep[key] = value.detach().cpu()
        ckpt_layers[idx] = keep

        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "softmax_alpha"):
            alpha_by_layer.append(attn.softmax_alpha.detach().cpu().clone())
        if attn is not None and hasattr(attn, "output_scale"):
            output_scale_by_layer.append(attn.output_scale.detach().cpu().clone())
        mlp = getattr(layer, "mlp", None)
        if include_mlp_output_scale and mlp is not None and hasattr(mlp, "mlp_output_scale"):
            mlp_output_scale_by_layer.append(mlp.mlp_output_scale.detach().cpu().clone())

    payload: Dict[str, Any] = {
        "meta": meta or {},
        "layers": ckpt_layers,
    }
    if alpha_by_layer:
        payload["softmax_alpha"] = torch.stack(alpha_by_layer)
    if output_scale_by_layer:
        payload["output_scale"] = torch.stack(output_scale_by_layer)
    if include_mlp_output_scale and mlp_output_scale_by_layer:
        payload["mlp_output_scale"] = torch.stack(mlp_output_scale_by_layer)
    torch.save(payload, path)


def iter_layer_x_mask_modules(layer) -> Iterable[nn.Module]:
    self_attn = getattr(layer, "self_attn", None)
    if self_attn is not None:
        for name in ("x_mask_in", "x_mask_out"):
            module = getattr(self_attn, name, None)
            if module is not None:
                yield module

    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        for name in ("x_mask_up", "x_mask_down"):
            module = getattr(mlp, name, None)
            if module is not None:
                yield module
