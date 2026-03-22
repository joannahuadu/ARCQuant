from __future__ import annotations

from typing import Iterable, Optional

import torch


def parse_layer_spec(spec) -> set[int]:
    if spec is None:
        return set()
    if isinstance(spec, (list, tuple, set)):
        return {int(x) for x in spec}
    text = str(spec).strip()
    if not text:
        return set()

    layers: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                start, end = end, start
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return layers


def iter_layer_x_mask_modules(layer) -> Iterable[torch.nn.Module]:
    """Yield all x-mask modules attached to an ARCQuant decoder layer."""
    self_attn = getattr(layer, "self_attn", None)
    if self_attn is not None:
        for name in ("x_mask_in", "x_mask_out"):
            xm = getattr(self_attn, name, None)
            if xm is not None:
                yield xm

    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        for name in ("x_mask_up", "x_mask_down"):
            xm = getattr(mlp, name, None)
            if xm is not None:
                yield xm

    # Mixtral-style MoE path: x-mask lives inside each expert MLP.
    block_sparse_moe = getattr(layer, "block_sparse_moe", None)
    if block_sparse_moe is not None:
        experts = getattr(block_sparse_moe, "experts", None)
        if experts is not None:
            for expert in experts:
                for name in ("x_mask_up", "x_mask_down"):
                    xm = getattr(expert, name, None)
                    if xm is not None:
                        yield xm


def set_layer_x_mask_alpha(layer, alpha: float) -> None:
    for xm in iter_layer_x_mask_modules(layer):
        xm.x_mask_alpha = float(alpha)


def set_layer_x_mask_eval_mode(layer, enable: bool) -> None:
    for xm in iter_layer_x_mask_modules(layer):
        if enable and hasattr(xm, "to_eval_mode"):
            xm.to_eval_mode()
        else:
            xm._eval_mode = False


def configure_x_mask_token_gate(
    model,
    *,
    use_x_mask: bool = False,
    x_mask_mode: str = "switch_top2_hard",
    x_mask_token_gate_mode: str = "static_all",
    x_mask_token_gate_deep_ratio: float = 0.5,
    x_mask_token_gate_deep_start: int = -1,
    x_mask_token_mlp_hidden: int = 0,
    x_mask_token_mlp_chunk_size: int = 1024,
    x_mask_token_mlp_shared: bool = True,
    x_mask_token_use_layer_scale: bool = True,
) -> None:
    """FlatQuant-compatible token-gate configurator for ARCQuant x-mask modules."""
    num_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if (
        not use_x_mask
        or "switch_top2" not in str(x_mask_mode)
        or x_mask_token_gate_mode == "static_all"
        or num_layers <= 0
    ):
        enabled_layers: set[int] = set()
    elif x_mask_token_gate_mode == "token_all":
        enabled_layers = set(range(num_layers))
    elif x_mask_token_gate_mode == "token_deep":
        if x_mask_token_gate_deep_start is not None and int(x_mask_token_gate_deep_start) >= 0:
            start = int(x_mask_token_gate_deep_start)
        else:
            start = int(num_layers * float(x_mask_token_gate_deep_ratio))
        start = max(0, min(start, num_layers))
        enabled_layers = set(range(start, num_layers))
    else:
        raise ValueError(f"Unknown x_mask_token_gate_mode: {x_mask_token_gate_mode}")

    layers = getattr(model.model, "layers", None)
    if layers is None:
        return

    for idx in range(num_layers):
        layer = layers[idx]
        shared_mlp = None
        enable = idx in enabled_layers
        for xm in iter_layer_x_mask_modules(layer):
            if not hasattr(xm, "x_mask_token_gate_enabled"):
                continue
            xm.x_mask_token_gate_enabled = enable
            xm.x_mask_token_mlp_hidden = int(x_mask_token_mlp_hidden)
            xm.x_mask_token_mlp_chunk_size = int(x_mask_token_mlp_chunk_size)
            xm.x_mask_token_use_layer_scale = bool(x_mask_token_use_layer_scale)

            scale = getattr(xm, "x_mask_token_scale", None)
            if scale is not None:
                scale.requires_grad_(False)
                if not x_mask_token_use_layer_scale:
                    with torch.no_grad():
                        scale.data.fill_(1.0)

            if not enable:
                continue

            if x_mask_token_mlp_shared:
                if shared_mlp is None:
                    if hasattr(xm, "_ensure_x_mask_token_mlp"):
                        shared_mlp = xm._ensure_x_mask_token_mlp()
                    else:
                        continue
                xm.x_mask_token_mlp = shared_mlp
                if hasattr(shared_mlp, "chunk_size"):
                    shared_mlp.chunk_size = int(x_mask_token_mlp_chunk_size)
            else:
                if hasattr(xm, "_ensure_x_mask_token_mlp"):
                    xm._ensure_x_mask_token_mlp()


def load_x_mask_checkpoint(model, ckpt_path: str) -> dict:
    """Load a `model/cali_x_mask.py` checkpoint into an ARCQuant model.

    The model must be built with `use_x_mask=True` so that x_mask modules exist.
    Returns the checkpoint `meta` dict (may be empty).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    layers_state = ckpt.get("layers", {}) if isinstance(ckpt, dict) else {}

    # Ensure token MLP modules exist before loading weights.
    if meta:
        configure_x_mask_token_gate(
            model,
            use_x_mask=True,
            x_mask_mode="switch_top2_hard",
            x_mask_token_gate_mode=meta.get("x_mask_token_gate_mode", "static_all"),
            x_mask_token_gate_deep_ratio=float(meta.get("x_mask_token_gate_deep_ratio", 0.5)),
            x_mask_token_gate_deep_start=int(meta.get("x_mask_token_gate_deep_start", -1)),
            x_mask_token_mlp_hidden=int(meta.get("x_mask_token_mlp_hidden", 0)),
            x_mask_token_mlp_chunk_size=int(meta.get("x_mask_token_mlp_chunk_size", 1024)),
            x_mask_token_mlp_shared=bool(meta.get("x_mask_token_mlp_shared", True)),
            x_mask_token_use_layer_scale=bool(meta.get("x_mask_token_use_layer_scale", True)),
        )

    layers = getattr(model.model, "layers", None)
    if layers is None:
        return meta

    for k, state in layers_state.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if idx < 0 or idx >= len(layers):
            continue
        layers[idx].load_state_dict(state, strict=False)

    return meta
