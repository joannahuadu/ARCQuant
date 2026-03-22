from __future__ import annotations

from typing import Iterable

import torch
from x_mask_utils import parse_layer_spec


def iter_attention_modules(model, *, skip_layers=None) -> Iterable[torch.nn.Module]:
    if hasattr(model, "module"):
        model = model.module
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        return
    skip_layers = parse_layer_spec(skip_layers)
    for layer_idx, layer in enumerate(layers):
        if layer_idx in skip_layers:
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "softmax_alpha"):
            yield attn


def _get_attention_layers(model):
    if hasattr(model, "module"):
        model = model.module
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        return []
    attn_layers = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "softmax_alpha"):
            attn_layers.append(attn)
        else:
            attn_layers.append(None)
    return attn_layers


def _coerce_alpha_tensor(alpha, n_layers: int | None = None) -> torch.Tensor:
    if isinstance(alpha, dict):
        for key in ("softmax_alpha", "alpha", "alpha_by_layer", "values"):
            if key in alpha:
                alpha = alpha[key]
                break
    alpha = torch.as_tensor(alpha, dtype=torch.float32)
    if alpha.dim() == 0:
        alpha = alpha.view(1, 1)
    elif alpha.dim() == 1:
        if n_layers is None or alpha.numel() == n_layers:
            alpha = alpha.view(-1, 1)
        else:
            alpha = alpha.view(1, -1)
    elif alpha.dim() != 2:
        raise ValueError(f"softmax alpha must be 0D/1D/2D, got shape {tuple(alpha.shape)}")
    return alpha


def set_model_softmax_alpha(model, alpha, *, skip_layers=None) -> torch.Tensor:
    attn_layers = _get_attention_layers(model)
    if not attn_layers or not any(attn is not None for attn in attn_layers):
        raise ValueError("no attention modules with softmax_alpha found")
    attn_modules = [attn for attn in attn_layers if attn is not None]
    skip_layers = parse_layer_spec(skip_layers)

    alpha = _coerce_alpha_tensor(alpha, n_layers=len(attn_layers))
    if alpha.shape[0] == 1 and len(attn_layers) > 1:
        alpha = alpha.expand(len(attn_layers), -1)
    if alpha.shape[0] != len(attn_layers):
        raise ValueError(f"softmax alpha layer mismatch: expected {len(attn_layers)}, got {alpha.shape[0]}")

    for layer_idx, attn in enumerate(attn_layers):
        if attn is None:
            continue
        if layer_idx in skip_layers:
            attn.softmax_alpha.fill_(1.0)
            continue
        layer_alpha = alpha[layer_idx]
        if layer_alpha.numel() == 1:
            layer_alpha = layer_alpha.expand_as(attn.softmax_alpha)
        if layer_alpha.numel() != attn.softmax_alpha.numel():
            raise ValueError(
                f"softmax alpha head mismatch at layer {layer_idx}: "
                f"expected {attn.softmax_alpha.numel()}, got {layer_alpha.numel()}"
            )
        attn.softmax_alpha.copy_(layer_alpha.to(device=attn.softmax_alpha.device, dtype=attn.softmax_alpha.dtype))
    return torch.stack(
        [
            (attn.softmax_alpha.detach().float().cpu() if attn is not None else torch.empty(0, dtype=torch.float32))
            for attn in attn_layers
            if attn is not None
        ],
        dim=0,
    )


def load_softmax_alpha_checkpoint(model, path: str, *, skip_layers=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    alpha = set_model_softmax_alpha(model, ckpt, skip_layers=skip_layers)
    return {"path": path, "shape": list(alpha.shape), **meta}
