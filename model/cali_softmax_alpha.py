from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from datautils import DEV, get_loaders
from main import get_llama, get_mixtral, get_qwen
from model_utils import reorder_model_llama, reorder_model_mixtral, reorder_model_qwen
from softmax_alpha_utils import set_model_softmax_alpha
from x_mask_utils import (
    iter_layer_x_mask_modules,
    load_x_mask_checkpoint,
    set_layer_x_mask_alpha,
    set_layer_x_mask_eval_mode,
)


def _parse_model_name(model_path: str) -> str:
    return model_path.split("/")[-2] if len(model_path.split("/")[-1]) == 0 else model_path.split("/")[-1]


def _resolve_model_family(model_path: str):
    lower = model_path.lower()
    if "llama" in lower:
        return get_llama, reorder_model_llama
    if "qwen" in lower:
        return get_qwen, reorder_model_qwen
    if "mixtral" in lower:
        return get_mixtral, reorder_model_mixtral
    raise ValueError(f"unsupported model family for path: {model_path}")


def _load_reorder_artifacts(model_name: str, dataset_name: str, metric: str):
    prefix = f"./saved/{model_name.lower()}"
    reorder_index = torch.load(f"{prefix}_reorder_index_{dataset_name}_{metric}.pt", weights_only=False)
    select_nums = torch.load(f"{prefix}_select_num_{dataset_name}_{metric}.pt", weights_only=False)
    return reorder_index, select_nums


def _build_quant_model(args, reorder_model_func, reorder_index, select_nums):
    model_builder, _ = _resolve_model_family(args.model)
    model = model_builder(args.model)
    model.eval()
    reorder_kwargs = {
        "device": DEV,
        "kv_cache": args.kv_cache,
        "reorder_index": reorder_index,
        "select_nums": select_nums,
        "quant_type": args.quant_type,
        "reorder_xw": not bool(args.no_xw_reorder),
        "use_x_mask": bool(args.use_x_mask),
        "x_mask_tau": float(args.x_mask_tau),
        "x_mask_alpha": float(args.x_mask_alpha),
        "x_mask_r_thr": None if float(args.x_mask_r_thr) < 0 else float(args.x_mask_r_thr),
    }
    if "llama" in args.model.lower():
        reorder_kwargs["rec"] = bool(args.rec)
    model = reorder_model_func(model, **reorder_kwargs)
    if args.use_x_mask and args.x_mask_ckpt:
        load_x_mask_checkpoint(model, args.x_mask_ckpt)
    if args.use_x_mask:
        x_mask_r_thr = None if float(args.x_mask_r_thr) < 0 else float(args.x_mask_r_thr)
        for layer in model.model.layers:
            set_layer_x_mask_alpha(layer, float(args.x_mask_alpha))
            if x_mask_r_thr is not None:
                for xm in iter_layer_x_mask_modules(layer):
                    xm.x_mask_r_thr = x_mask_r_thr
            if args.x_mask_eval_hard:
                set_layer_x_mask_eval_mode(layer, True)
    model.to(DEV)
    model.eval()
    return model


def _normalized_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    valid_counts = torch.isfinite(logits).sum(dim=-1).clamp_min(2).to(dtype=ent.dtype)
    return ent / valid_counts.log()


class AlphaCalibrationRecorder:
    def __init__(
        self,
        *,
        model,
        mode: str,
        seed: int,
        sample_rows: int,
        max_rows_per_head: int,
        alpha_grid: torch.Tensor | None = None,
        targets: dict[int, dict[int, torch.Tensor]] | None = None,
    ) -> None:
        self.model = model
        self.mode = mode
        self.seed = int(seed)
        self.sample_rows = int(sample_rows)
        self.max_rows_per_head = int(max_rows_per_head)
        self.alpha_grid = alpha_grid
        self.targets = targets

        self._current_layer = None
        self._layer_handles = []
        self._orig_sdpa = None
        self._call_idx = 0

        self.target_entropy = defaultdict(lambda: defaultdict(list))
        self.loss_sum = defaultdict(dict)
        self.loss_count = defaultdict(dict)
        self.target_cursor = defaultdict(lambda: defaultdict(int))

    def _install_layer_hooks(self):
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        layers = base_model.model.layers if hasattr(base_model, "model") else base_model.layers

        def make_pre(i):
            def _pre(_module, _inputs):
                self._current_layer = i
            return _pre

        def make_post(_i):
            def _post(_module, _inputs, _output):
                self._current_layer = None
            return _post

        for i, layer in enumerate(layers):
            self._layer_handles.append(layer.register_forward_pre_hook(make_pre(i)))
            self._layer_handles.append(layer.register_forward_hook(make_post(i)))

    def _sample_mask_rows(self, attn_mask, b_idx, h_idx, q_idx, k_len, start_k):
        if attn_mask is None or not torch.is_tensor(attn_mask):
            return None
        rows = None
        try:
            if attn_mask.dim() == 2:
                rows = attn_mask[q_idx.clamp_max(attn_mask.shape[0] - 1), :]
            elif attn_mask.dim() == 3:
                rows = attn_mask[
                    b_idx.clamp_max(attn_mask.shape[0] - 1),
                    q_idx.clamp_max(attn_mask.shape[1] - 1),
                    :,
                ]
            elif attn_mask.dim() == 4:
                rows = attn_mask[
                    b_idx.clamp_max(attn_mask.shape[0] - 1),
                    h_idx.clamp_max(attn_mask.shape[1] - 1),
                    q_idx.clamp_max(attn_mask.shape[2] - 1),
                    :,
                ]
        except Exception:
            return None
        if rows is None or rows.shape[-1] <= start_k:
            return None
        rows = rows[..., start_k:]
        if rows.shape[-1] < k_len:
            return None
        return rows[..., :k_len]

    def _sample_row_indices(self, total_rows: int) -> torch.Tensor:
        n_rows = min(max(self.sample_rows, 1), total_rows)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self._call_idx * 104729 + int(self._current_layer or 0) * 1009)
        return torch.randint(0, total_rows, (n_rows,), generator=generator)

    def _observe_sdpa(self, query, key, attn_mask=None, is_causal=False, scale=None):
        if self._current_layer is None:
            return
        if not (torch.is_tensor(query) and torch.is_tensor(key)):
            return
        if query.dim() != 4 or key.dim() != 4 or query.shape[-1] != key.shape[-1]:
            return

        bsz, n_heads, q_len, head_dim = query.shape
        total_rows = int(bsz * n_heads * q_len)
        if total_rows <= 0:
            return

        ridx = self._sample_row_indices(total_rows).to(device=query.device)
        q_idx = ridx % q_len
        tmp = ridx // q_len
        h_idx = tmp % n_heads
        b_idx = tmp // n_heads

        q_rows = query[b_idx, h_idx, q_idx, :].float()
        k_rows = key[b_idx, h_idx, :, :].float()
        logits = torch.einsum("nd,nkd->nk", q_rows, k_rows)
        scale_value = float(scale) if scale is not None else (1.0 / math.sqrt(float(head_dim)))
        logits = logits * scale_value
        k_len = logits.shape[-1]

        mask_rows = self._sample_mask_rows(attn_mask, b_idx, h_idx, q_idx, k_len, 0)
        if mask_rows is not None:
            mask_rows = mask_rows.to(logits.device)
            if mask_rows.dtype == torch.bool:
                logits = logits.masked_fill(~mask_rows, float("-inf"))
            else:
                logits = logits + mask_rows.to(dtype=logits.dtype)

        if is_causal:
            offset = max(int(key.shape[-2]) - int(query.shape[-2]), 0)
            q_pos = q_idx + offset
            cols = torch.arange(k_len, device=logits.device).view(1, -1)
            logits = logits.masked_fill(cols > q_pos.view(-1, 1), float("-inf"))

        valid = torch.isfinite(logits).any(dim=-1)
        if not bool(valid.any()):
            return

        logits = logits[valid]
        h_idx = h_idx[valid]

        layer = int(self._current_layer)
        for head in h_idx.unique(sorted=True).tolist():
            head = int(head)
            sel = h_idx == head
            logits_head = logits[sel]
            if logits_head.numel() == 0:
                continue

            if self.mode == "bf16":
                remain = self.max_rows_per_head - len(self.target_entropy[layer][head])
                if remain <= 0:
                    continue
                logits_head = logits_head[:remain]
                ent = _normalized_entropy_from_logits(logits_head).cpu()
                self.target_entropy[layer][head].append(ent)
                continue

            target = self.targets.get(layer, {}).get(head)
            if target is None or target.numel() == 0:
                continue
            cursor = self.target_cursor[layer][head]
            remain = int(target.numel() - cursor)
            if remain <= 0:
                continue
            logits_head = logits_head[:remain]
            target_slice = target[cursor : cursor + logits_head.shape[0]].to(device=logits_head.device, dtype=logits_head.dtype)
            self.target_cursor[layer][head] += logits_head.shape[0]

            scaled_logits = self.alpha_grid.view(-1, 1, 1).to(logits_head.device, logits_head.dtype) * logits_head.unsqueeze(0)
            ent_grid = _normalized_entropy_from_logits(scaled_logits)
            diff = ent_grid - target_slice.unsqueeze(0)
            loss = (diff * diff).sum(dim=-1).cpu()

            if head not in self.loss_sum[layer]:
                self.loss_sum[layer][head] = loss
                self.loss_count[layer][head] = logits_head.shape[0]
            else:
                self.loss_sum[layer][head] += loss
                self.loss_count[layer][head] += logits_head.shape[0]

    @contextmanager
    def patch(self):
        self._install_layer_hooks()
        self._orig_sdpa = F.scaled_dot_product_attention

        def _wrapped_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
            out = self._orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
            try:
                self._observe_sdpa(query, key, attn_mask=attn_mask, is_causal=is_causal, scale=scale)
            finally:
                self._call_idx += 1
            return out

        F.scaled_dot_product_attention = _wrapped_sdpa
        try:
            yield self
        finally:
            F.scaled_dot_product_attention = self._orig_sdpa
            for h in self._layer_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._layer_handles = []


def _collect_targets(model, loader, args):
    recorder = AlphaCalibrationRecorder(
        model=model,
        mode="bf16",
        seed=args.seed,
        sample_rows=args.sample_rows,
        max_rows_per_head=args.max_rows_per_head,
    )
    with torch.no_grad(), recorder.patch():
        for idx, (inp, _tar) in enumerate(loader):
            if idx >= args.nsamples:
                break
            model(inp.to(DEV))
    result = {}
    for layer, by_head in recorder.target_entropy.items():
        result[layer] = {}
        for head, chunks in by_head.items():
            result[layer][head] = torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.float32)
    return result


def _fit_alpha(model, loader, args, targets):
    grid = torch.arange(args.alpha_min, args.alpha_max + 0.5 * args.alpha_step, args.alpha_step, dtype=torch.float32)
    recorder = AlphaCalibrationRecorder(
        model=model,
        mode="fit",
        seed=args.seed,
        sample_rows=args.sample_rows,
        max_rows_per_head=args.max_rows_per_head,
        alpha_grid=grid,
        targets=targets,
    )
    with torch.no_grad(), recorder.patch():
        for idx, (inp, _tar) in enumerate(loader):
            if idx >= args.nsamples:
                break
            model(inp.to(DEV))

    n_layers = len(model.model.layers)
    num_heads = model.model.layers[0].self_attn.num_heads
    alpha = torch.ones((n_layers, num_heads), dtype=torch.float32)
    counts = torch.zeros((n_layers, num_heads), dtype=torch.int32)

    for layer in range(n_layers):
        for head in range(num_heads):
            if head not in recorder.loss_sum.get(layer, {}):
                continue
            loss = recorder.loss_sum[layer][head]
            best = int(torch.argmin(loss).item())
            alpha[layer, head] = grid[best]
            counts[layer, head] = int(recorder.loss_count[layer][head])
    return alpha, counts, grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--act_sort_metric", type=str, default="max", choices=["mean", "frobenius", "hessian", "max"])
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "pile", "humaneval"])
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--kv_cache", action="store_true")
    parser.add_argument("--quant_type", type=str, default="NVFP4", choices=["NVFP4", "MXFP4", "INT4", "HiF4"])
    parser.add_argument("--no_xw_reorder", action="store_true")
    parser.add_argument("--use_x_mask", action="store_true")
    parser.add_argument("--x_mask_ckpt", type=str, default=None)
    parser.add_argument("--x_mask_tau", type=float, default=1.0)
    parser.add_argument("--x_mask_alpha", type=float, default=1.0)
    parser.add_argument("--x_mask_r_thr", type=float, default=-1.0)
    parser.add_argument("--x_mask_eval_hard", action="store_true")
    parser.add_argument("--rec", action="store_true")
    parser.add_argument("--sample_rows", type=int, default=256)
    parser.add_argument("--max_rows_per_head", type=int, default=64)
    parser.add_argument("--alpha_min", type=float, default=0.5)
    parser.add_argument("--alpha_max", type=float, default=1.5)
    parser.add_argument("--alpha_step", type=float, default=0.005)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    if args.alpha_step <= 0:
        raise ValueError("--alpha_step must be > 0")
    if args.alpha_max <= args.alpha_min:
        raise ValueError("--alpha_max must be > --alpha_min")

    model_name = _parse_model_name(args.model)
    dataset_name = args.dataset.lower()
    model_builder, reorder_model_func = _resolve_model_family(args.model)
    reorder_index, select_nums = _load_reorder_artifacts(model_name, dataset_name, args.act_sort_metric)
    loader, _testenc, _tokenizer = get_loaders(dataset_name, nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, model=args.model)

    print("Collecting bf16 target entropy...")
    bf16_model = model_builder(args.model).to(DEV)
    bf16_model.eval()
    targets = _collect_targets(bf16_model, loader, args)
    del bf16_model
    torch.cuda.empty_cache()

    print("Building quantized model and fitting softmax alpha...")
    quant_model = _build_quant_model(args, reorder_model_func, reorder_index, select_nums)
    alpha, counts, grid = _fit_alpha(quant_model, loader, args, targets)
    set_model_softmax_alpha(quant_model, alpha)

    output_path = args.output_path
    if output_path is None:
        output_dir = ROOT_DIR / "outputs" / model_name / args.quant_type
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"softmax_alpha_{dataset_name}_{args.act_sort_metric}.pt"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "softmax_alpha": alpha.cpu(),
            "counts": counts.cpu(),
            "meta": {
                "model": args.model,
                "dataset": args.dataset,
                "nsamples": args.nsamples,
                "seqlen": args.seqlen,
                "quant_type": args.quant_type,
                "sample_rows": args.sample_rows,
                "max_rows_per_head": args.max_rows_per_head,
                "alpha_min": args.alpha_min,
                "alpha_max": args.alpha_max,
                "alpha_step": args.alpha_step,
                "grid_size": int(grid.numel()),
            },
        },
        output_path,
    )
    print(f"Saved softmax alpha to: {output_path}")
    print(f"alpha mean={alpha.mean().item():.6f}, min={alpha.min().item():.6f}, max={alpha.max().item():.6f}")
    print(f"rows used per head: min={counts.min().item()}, max={counts.max().item()}, mean={counts.float().mean().item():.2f}")


if __name__ == "__main__":
    main()
