from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from x_mask_utils import parse_layer_spec


class LowRankResidual(nn.Module):
    """Lightweight residual adapter: y = x + B(A(x))."""

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        hidden_size = int(hidden_size)
        rank = int(rank)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.hidden_size = hidden_size
        self.rank = rank
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x.float()))
        return out.to(dtype=x.dtype)


def iter_low_rank_modules(model) -> Iterable[nn.Module]:
    if hasattr(model, "module"):
        model = model.module
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        return

    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            mod = getattr(attn, "output_low_rank", None)
            if mod is not None:
                yield mod
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            mod = getattr(mlp, "output_low_rank", None)
            if mod is not None:
                yield mod


def configure_model_low_rank(
    model,
    *,
    attn_layers=None,
    attn_rank: int = 0,
    mlp_layers=None,
    mlp_rank: int = 0,
) -> None:
    if hasattr(model, "module"):
        model = model.module
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        return

    attn_layers = parse_layer_spec(attn_layers)
    mlp_layers = parse_layer_spec(mlp_layers)
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "configure_output_low_rank"):
            attn.configure_output_low_rank(attn_rank if idx in attn_layers else 0)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "configure_output_low_rank"):
            mlp.configure_output_low_rank(mlp_rank if idx in mlp_layers else 0)

