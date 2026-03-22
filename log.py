import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
FLATQUANT_DIR = ROOT_DIR.parent / "FlatQuant"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(FLATQUANT_DIR) not in sys.path:
    sys.path.insert(0, str(FLATQUANT_DIR))

from datautils import DEV, get_loaders
from eval import eval_ppl
from model_utils import reorder_model_llama, reorder_model_mixtral, reorder_model_qwen
from qLinearLayer import QLinearLayer
from x_mask_utils import (
    iter_layer_x_mask_modules,
    load_x_mask_checkpoint,
    parse_layer_spec,
    set_layer_x_mask_alpha,
    set_layer_x_mask_eval_mode,
)
from softmax_alpha_utils import load_softmax_alpha_checkpoint


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("arcquant_log")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class SoftmaxStatsScope:
    def __init__(self, args, logger, model=None) -> None:
        self.args = args
        self.logger = logger
        self.model = model
        self.softmax_stats = None
        self._ctx = nullcontext()
        self._layer_hook_handles = []
        self._orig_sdpa = None
        self._sdpa_rows = int(getattr(args, "softmax_stats_sdpa_rows", 256))
        self._sdpa_k_cap = int(getattr(args, "softmax_stats_sdpa_k_cap", 0))

        if not getattr(args, "softmax_stats", False):
            return

        from flatquant.softmax_stats import SoftmaxStatsCollector, SoftmaxStatsConfig

        self.softmax_stats = SoftmaxStatsCollector(
            SoftmaxStatsConfig(
                sample_per_call=int(getattr(args, "softmax_stats_sample", 262144)),
                max_calls=int(getattr(args, "softmax_stats_max_calls", 0)),
                bins_linear=int(getattr(args, "softmax_stats_bins", 200)),
                bins_log10=int(getattr(args, "softmax_stats_log_bins", 240)),
                log10_min=float(getattr(args, "softmax_stats_log_min", -12.0)),
                only_last_dim_min=int(getattr(args, "softmax_stats_min_kv", 16)),
                entropy_rows_per_call=int(getattr(args, "softmax_stats_entropy_rows", 8192)),
                entropy_bins=int(getattr(args, "softmax_stats_entropy_bins", 200)),
                per_layer=bool(getattr(args, "softmax_stats_per_layer", False)),
                per_head=bool(getattr(args, "softmax_stats_per_head", False)),
                head_dim=int(getattr(args, "softmax_stats_head_dim", 1)),
                row_std_rows_per_call=int(getattr(args, "softmax_stats_row_std_rows", 2048)),
                top1_lag_rows_per_call=int(getattr(args, "softmax_stats_top1_lag_rows", 2048)),
                top1_lag_bins=int(getattr(args, "softmax_stats_top1_lag_bins", 512)),
                top1_lag_max=int(getattr(args, "softmax_stats_top1_lag_max", 8192)),
            ),
            logger=logger,
        )
        self._ctx = self._patch()

        logger.info(
            "[softmax_stats] enabled: "
            f"sample_per_call={self.softmax_stats.config.sample_per_call}, "
            f"max_calls={self.softmax_stats.config.max_calls}, "
            f"bins={self.softmax_stats.config.bins_linear}, "
            f"log_bins={self.softmax_stats.config.bins_log10}, "
            f"log10_min={self.softmax_stats.config.log10_min}, "
            f"entropy_rows={self.softmax_stats.config.entropy_rows_per_call}, "
            f"entropy_bins={self.softmax_stats.config.entropy_bins}, "
            f"per_layer={self.softmax_stats.config.per_layer}, "
            f"per_head={self.softmax_stats.config.per_head}, "
            f"head_dim={self.softmax_stats.config.head_dim}, "
            f"row_std_rows={self.softmax_stats.config.row_std_rows_per_call}, "
            f"top1_lag_rows={self.softmax_stats.config.top1_lag_rows_per_call}, "
            f"top1_lag_bins={self.softmax_stats.config.top1_lag_bins}, "
            f"top1_lag_max={self.softmax_stats.config.top1_lag_max}, "
            f"sdpa_rows={self._sdpa_rows}, "
            f"sdpa_k_cap={self._sdpa_k_cap}"
        )

    def _install_layer_hooks(self) -> None:
        if not (
            self.softmax_stats
            and (self.softmax_stats.config.per_layer or self.softmax_stats.config.per_head)
        ):
            return
        model = self.model
        if model is None:
            self.logger.warning("[softmax_stats] per-layer/per-head requested but model is None; skip layer hooks.")
            return
        if hasattr(model, "module"):
            model = model.module
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        if layers is None:
            self.logger.warning("[softmax_stats] per-layer/per-head requested but could not find model.layers; skip layer hooks.")
            return

        collector = self.softmax_stats

        def make_pre(i: int):
            def _pre(_module, _inputs):
                collector.push_layer(i)
            return _pre

        def make_post(_i: int):
            def _post(_module, _inputs, _output):
                collector.pop_layer()
            return _post

        for i, layer in enumerate(layers):
            self._layer_hook_handles.append(layer.register_forward_pre_hook(make_pre(i)))
            self._layer_hook_handles.append(layer.register_forward_hook(make_post(i)))
        self.logger.info(f"[softmax_stats] layer context enabled on {len(layers)} layers.")

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

    @torch.no_grad()
    def _observe_sdpa(self, query, key, attn_mask=None, is_causal=False, scale=None):
        if self.softmax_stats is None:
            return
        if not (torch.is_tensor(query) and torch.is_tensor(key)):
            return
        if query.dim() != 4 or key.dim() != 4:
            return
        if query.shape[-1] != key.shape[-1]:
            return
        if key.shape[-2] < self.softmax_stats.config.only_last_dim_min:
            return

        bsz, n_heads, q_len, head_dim = query.shape
        total_rows = int(bsz * n_heads * q_len)
        if total_rows <= 0:
            return

        n_rows = min(max(self._sdpa_rows, 1), total_rows)
        ridx = torch.randint(0, total_rows, (n_rows,), device=query.device)
        q_idx = ridx % q_len
        tmp = ridx // q_len
        h_idx = tmp % n_heads
        b_idx = tmp // n_heads

        q_rows = query[b_idx, h_idx, q_idx, :].float()
        k_rows = key[b_idx, h_idx, :, :].float()

        start_k = 0
        if self._sdpa_k_cap > 0 and k_rows.shape[1] > self._sdpa_k_cap:
            start_k = int(k_rows.shape[1] - self._sdpa_k_cap)
            k_rows = k_rows[:, start_k:, :]

        logits = torch.einsum("nd,nkd->nk", q_rows, k_rows)
        scale_value = float(scale) if scale is not None else (1.0 / (float(head_dim) ** 0.5))
        logits = logits * scale_value
        k_len = logits.shape[-1]

        mask_rows = self._sample_mask_rows(attn_mask, b_idx, h_idx, q_idx, k_len, start_k)
        if mask_rows is not None:
            mask_rows = mask_rows.to(logits.device)
            if mask_rows.dtype == torch.bool:
                logits = logits.masked_fill(~mask_rows, float("-inf"))
            else:
                logits = logits + mask_rows.to(dtype=logits.dtype)

        if is_causal:
            offset = max(int(key.shape[-2]) - int(query.shape[-2]), 0)
            q_pos = q_idx + offset - int(start_k)
            cols = torch.arange(k_len, device=logits.device).view(1, -1)
            logits = logits.masked_fill(cols > q_pos.view(-1, 1), float("-inf"))

        valid = torch.isfinite(logits).any(dim=-1)
        if not bool(valid.any()):
            return
        probs = torch.softmax(logits[valid], dim=-1)
        probs = probs[torch.isfinite(probs).all(dim=-1)]
        if probs.numel() == 0:
            return
        self.softmax_stats.observe(probs.view(1, 1, probs.shape[0], probs.shape[1]), dim=-1)

    @contextmanager
    def _patch(self):
        with self.softmax_stats.patch():
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
                except Exception:
                    self.softmax_stats.n_skipped += 1
                return out

            F.scaled_dot_product_attention = _wrapped_sdpa
            try:
                yield self.softmax_stats
            finally:
                F.scaled_dot_product_attention = self._orig_sdpa
                self._orig_sdpa = None

    def __enter__(self):
        self._ctx.__enter__()
        self._install_layer_hooks()
        return self.softmax_stats

    def __exit__(self, exc_type, exc, tb):
        for h in self._layer_hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._layer_hook_handles = []
        suppress = self._ctx.__exit__(exc_type, exc, tb)
        if exc_type is None and self.softmax_stats is not None:
            save_prefix = (
                self.args.softmax_stats_save_path
                if getattr(self.args, "softmax_stats_save_path", None)
                else os.path.join(self.args.exp_dir, "softmax_stats")
            )
            pt_path, json_path = self.softmax_stats.save(save_prefix)
            self.logger.info(f"[softmax_stats] saved: {pt_path} / {json_path}")
        return suppress


def _log_xq_stats(
    x_q,
    tag,
    logger,
    logged,
    out_dir=None,
    bins=120,
    max_points=200_000,
):
    if tag in logged:
        return
    with torch.no_grad():
        total = x_q.numel()
        if total == 0:
            return
        zeros = (x_q == 0)
        zero_ratio = zeros.float().mean().item()
        if x_q.shape[-1] % 4 == 0:
            g = x_q.reshape(-1, x_q.shape[-1] // 4, 4)
            zeros_g = (g == 0).sum(dim=-1)
            two_zeros_ratio = (zeros_g >= 2).float().mean().item()
            logger.info(f"[{tag}] zero_ratio={zero_ratio:.6f}, two_zeros_ratio={two_zeros_ratio:.6f}")
        else:
            logger.info(f"[{tag}] zero_ratio={zero_ratio:.6f} (last dim not divisible by 4)")

        if out_dir is not None:
            try:
                import numpy as np
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except Exception as exc:
                logger.warning(f"[{tag}] skip plot (matplotlib import failed): {exc}")
            else:
                os.makedirs(out_dir, exist_ok=True)
                flat = x_q.detach().float().view(-1)
                if flat.numel() > max_points:
                    idx = torch.randint(0, flat.numel(), (max_points,), device=flat.device)
                    flat = flat[idx]
                data = flat.cpu().numpy()
                mean = float(data.mean())
                std = float(data.std())

                fig = plt.figure(figsize=(5.2, 3.4))
                plt.hist(data, bins=bins, density=True, alpha=0.7, color="#8ecae6")
                if std > 0:
                    xs = np.linspace(mean - 4 * std, mean + 4 * std, 200)
                    pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mean) / std) ** 2)
                    plt.plot(xs, pdf, "--", color="#023047", linewidth=1.5, label="Normal fit")
                    plt.legend()
                plt.title(tag)
                plt.xlabel("Activation Value")
                plt.ylabel("Density")
                plt.tight_layout()

                safe_tag = tag.replace("/", "_").replace(".", "_")
                out_path = os.path.join(out_dir, f"{safe_tag}_dist.png")
                plt.savefig(out_path, dpi=200)
                plt.close(fig)
                logger.info(f"[{tag}] saved activation distribution: {out_path}")
        logged.add(tag)


def _is_target_module(name: str) -> bool:
    suffixes = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "block_sparse_moe.experts.0.w1",
        "block_sparse_moe.experts.0.w2",
        "block_sparse_moe.experts.0.w3",
    )
    if name.endswith(suffixes):
        return True
    return ".block_sparse_moe.experts." in name and (
        name.endswith(".w1") or name.endswith(".w2") or name.endswith(".w3")
    )


def _register_log_hooks(model, logger, args):
    logged = set()
    plot_dir = os.path.join(args.exp_dir, "act_dist")

    for name, module in model.named_modules():
        if not isinstance(module, QLinearLayer):
            continue
        if not _is_target_module(name):
            continue

        def pre_hook(_module, inputs, _name=name):
            if not inputs:
                return
            packed = inputs[0]
            if not isinstance(packed, (tuple, list)) or not packed:
                return
            x_q = packed[0]
            if not torch.is_tensor(x_q):
                return
            _log_xq_stats(x_q, tag=_name, logger=logger, logged=logged, out_dir=plot_dir)

        module.register_forward_pre_hook(pre_hook)

    return logged


def get_llama(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    return model


def get_qwen(model):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
    return model


def get_mixtral(model):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--act_sort_metric", type=str, default="max", choices=["mean", "frobenius", "hessian", "max"])
    parser.add_argument("--kv_cache", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"])
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--lm_eval_batch_size", type=int, default=16)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "pile", "humaneval"])
    parser.add_argument("--quant_type", type=str, default="NVFP4", choices=["NVFP4", "MXFP4", "INT4", "HiF4"])
    parser.add_argument("--no_xw_reorder", action="store_true")
    parser.add_argument("--use_x_mask", action="store_true")
    parser.add_argument("--x_mask_ckpt", type=str, default=None)
    parser.add_argument("--x_mask_tau", type=float, default=1.0)
    parser.add_argument("--x_mask_alpha", type=float, default=1.0)
    parser.add_argument("--x_mask_skip_layers", type=str, default="", help="Comma/range list of layer ids to skip x-mask, e.g. '0,1,8-15'.")
    parser.add_argument("--x_mask_r_thr", type=float, default=-1.0)
    parser.add_argument("--x_mask_eval_hard", action="store_true")
    parser.add_argument(
        "--softmax_alpha_ckpt",
        type=str,
        default=None,
        help="Load per-layer/per-head softmax alpha checkpoint produced by `python model/cali_softmax_alpha.py ...`.",
    )
    parser.add_argument(
        "--softmax_alpha_skip_layers",
        type=str,
        default="",
        help="Comma/range list of layer ids to skip softmax alpha, e.g. '0,1,8-15'.",
    )
    parser.add_argument(
        "--rec",
        action="store_true",
        help="Preserve pre-mask reconstruction channels for x_rec when x-mask is enabled.",
    )
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--softmax_stats", action="store_true", default=False)
    parser.add_argument("--softmax_stats_sample", type=int, default=262144)
    parser.add_argument("--softmax_stats_max_calls", type=int, default=0)
    parser.add_argument("--softmax_stats_bins", type=int, default=200)
    parser.add_argument("--softmax_stats_log_bins", type=int, default=240)
    parser.add_argument("--softmax_stats_log_min", type=float, default=-12.0)
    parser.add_argument("--softmax_stats_min_kv", type=int, default=16)
    parser.add_argument("--softmax_stats_entropy_rows", type=int, default=8192)
    parser.add_argument("--softmax_stats_entropy_bins", type=int, default=200)
    parser.add_argument("--softmax_stats_per_layer", action="store_true", default=False)
    parser.add_argument("--softmax_stats_per_head", action="store_true", default=False)
    parser.add_argument("--softmax_stats_head_dim", type=int, default=1)
    parser.add_argument("--softmax_stats_row_std_rows", type=int, default=2048)
    parser.add_argument("--softmax_stats_top1_lag_rows", type=int, default=2048)
    parser.add_argument("--softmax_stats_top1_lag_bins", type=int, default=512)
    parser.add_argument("--softmax_stats_top1_lag_max", type=int, default=8192)
    parser.add_argument("--softmax_stats_sdpa_rows", type=int, default=256)
    parser.add_argument("--softmax_stats_sdpa_k_cap", type=int, default=0)
    parser.add_argument("--softmax_stats_save_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model.split("/")[-2] if len(args.model.split("/")[-1]) == 0 else args.model.split("/")[-1]
    if args.exp_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_dir = os.path.join("outputs", f"log_{model_name}_{stamp}")
    os.makedirs(args.exp_dir, exist_ok=True)
    logger = build_logger(Path(args.exp_dir) / "run.log")

    if "llama" in args.model.lower():
        model = get_llama(args.model)
        reorder_model_func = reorder_model_llama
    elif "qwen" in args.model.lower():
        model = get_qwen(args.model)
        reorder_model_func = reorder_model_qwen
    elif "mixtral" in args.model.lower():
        model = get_mixtral(args.model)
        reorder_model_func = reorder_model_mixtral
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model.eval()

    dataset_name = args.dataset.lower()
    index_filename = f"./saved/{model_name.lower()}_reorder_index_{dataset_name}_{args.act_sort_metric}.pt"
    select_num_filename = f"./saved/{model_name.lower()}_select_num_{dataset_name}_{args.act_sort_metric}.pt"
    act_scales_filename = f"./saved/{model_name.lower()}_act_scales_{dataset_name}_{args.act_sort_metric}.pt"

    assert os.path.isfile(index_filename), "reorder index file not found."

    logger.info("Loading cached reording index from disk...")
    reorder_index = torch.load(index_filename, weights_only=False)
    select_nums = torch.load(select_num_filename, weights_only=False)
    _ = torch.load(act_scales_filename, weights_only=False)

    torch.cuda.reset_max_memory_allocated()
    logger.info("Reordering model...")
    start_time = time.time()
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
        "x_mask_skip_layers": args.x_mask_skip_layers,
        "x_mask_r_thr": None if float(args.x_mask_r_thr) < 0 else float(args.x_mask_r_thr),
    }
    if "llama" in args.model.lower():
        reorder_kwargs["rec"] = bool(args.rec)
    model = reorder_model_func(
        model,
        **reorder_kwargs,
    )
    model.eval()
    end_time = time.time()
    peak_memory = torch.cuda.max_memory_allocated()

    if args.use_x_mask and args.x_mask_ckpt:
        meta = load_x_mask_checkpoint(model, args.x_mask_ckpt)
        if meta:
            logger.info(f"Loaded x-mask ckpt meta: {meta}")

    if args.softmax_alpha_ckpt:
        meta = load_softmax_alpha_checkpoint(
            model,
            args.softmax_alpha_ckpt,
            skip_layers=args.softmax_alpha_skip_layers,
        )
        if meta:
            logger.info(f"Loaded softmax alpha ckpt meta: {meta}")

    if args.use_x_mask:
        skip_layers = parse_layer_spec(args.x_mask_skip_layers)
        x_mask_r_thr = None if float(args.x_mask_r_thr) < 0 else float(args.x_mask_r_thr)
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx in skip_layers:
                continue
            set_layer_x_mask_alpha(layer, float(args.x_mask_alpha))
            if x_mask_r_thr is not None:
                for xm in iter_layer_x_mask_modules(layer):
                    xm.x_mask_r_thr = x_mask_r_thr
            # if args.x_mask_eval_hard:
                set_layer_x_mask_eval_mode(layer, True)

    logger.info(f"Quantized Model Size: {peak_memory / (1024 * 1024 * 1024):.2f} GB")
    logger.info(f"Quantized Type is: {args.quant_type}")
    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")

    model.to(DEV)
    _register_log_hooks(model, logger, args)

    tokenizer = None
    # with SoftmaxStatsScope(args, logger, model=model):
    if args.eval_ppl:
        for dataset in ["wikitext2"]:
            _, testloader, tokenizer = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=2048)
            logger.info(f"Evaluating {dataset} ...")
            ppl = eval_ppl(model, testloader)
            logger.info(f"Result,{dataset},{ppl:.3f}")

    if args.tasks is not None:
        import random
        import lm_eval
        import numpy as np
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import initialize_tasks
        from transformers import AutoTokenizer

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)

        initialize_tasks()
        task_patterns = []
        for item in args.tasks:
            task_patterns.extend([x.strip() for x in str(item).split(",") if x.strip()])

        task_names = sorted(set(lm_eval_utils.pattern_match(task_patterns, lm_eval.tasks.ALL_TASKS)))
        results_by_task = {}
        with SoftmaxStatsScope(args, logger, model=model):
            for task_name in task_names:
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                model.eval()
                hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)
                result = lm_eval.simple_evaluate(
                    hflm,
                    tasks=[task_name],
                    num_fewshot=args.num_fewshot,
                    batch_size=args.lm_eval_batch_size,
                )
                results_by_task[task_name] = result.get("results", {}).get(task_name, {})
        for task, metrics in results_by_task.items():
            logger.info(task)
            for k, v in metrics.items():
                if "stderr" not in k:
                    logger.info(f"  {k}: {v}")

        if args.output_file:
            import json

            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results_by_task, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
