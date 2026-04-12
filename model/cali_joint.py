"""Joint calibration of x_mask gates and softmax_alpha.

Based on cali_x_mask.py but jointly trains per-layer softmax_alpha alongside
the sparsity gate logits. This avoids the sequential calibration conflict where
independently optimized gates and alpha values interfere with each other.

Key difference from cali_x_mask.py:
  - softmax_alpha (per-head attention temperature) is added to trainable params
  - Optional alpha regularization: lambda * (alpha - 1)^2
  - Saves both x_mask state and softmax_alpha in a single checkpoint
"""
import argparse
import gc
import math
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import logging
from termcolor import colored
from datetime import datetime
import pprint
from datautils import DEV, get_loaders
from model_utils import reorder_model_llama, reorder_model_mixtral, reorder_model_qwen
from x_mask_utils import iter_layer_x_mask_modules, set_layer_x_mask_alpha, set_layer_x_mask_eval_mode, configure_x_mask_token_gate, parse_layer_spec


def create_logger(exp_dir, dist_rank=0, name=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    log_file = os.path.join(exp_dir, f'log_rank{dist_rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def _get_model(model_path: str):
    model_path_l = model_path.lower()
    if "llama" in model_path_l:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        return model, reorder_model_llama
    if "qwen" in model_path_l:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
        return model, reorder_model_qwen
    if "mixtral" in model_path_l:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
        return model, reorder_model_mixtral
    raise ValueError(f"Unknown model type from path: {model_path}")


def _model_name_from_path(model_path: str) -> str:
    parts = [p for p in model_path.split("/") if p]
    return parts[-1] if parts else "model"


def _unique_params(params):
    seen = set()
    out = []
    for p in params:
        if p is None:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=[
                            "wikitext2",
                            "c4",
                            "pile",
                            "humaneval",
                            "arc_mix",
                            "wikitext2_c4_mix_1to1",
                            "wikitext2_c4_mix_3to1",
                            "wikitext2_c4_zh_mix_1to1to1",
                            "wikitext2_c4_zh_mix_2to2to1",
                            "wikitext2_c4_zh_mix_3to3to2",
                        ])
    parser.add_argument("--act_sort_metric", type=str, default="max",
                        choices=["mean", "frobenius", "hessian", "max"])
    parser.add_argument("--quant_type", type=str, default="NVFP4",
                        choices=["NVFP4", "MXFP4", "INT4", "HiF4"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--exp_name", type=str, default="joint")

    # x-mask config
    parser.add_argument("--x_mask_tau", type=float, default=1.0)
    parser.add_argument("--x_mask_alpha", type=float, default=1.0)
    parser.add_argument("--x_mask_skip_layers", type=str, default="")
    parser.add_argument("--x_mask_r_thr", type=float, default=-1.0)
    parser.add_argument("--x_mask_train_hard_r_thr", action="store_true",
                        help="Enable training-time hard switch for x_mask_r_thr using STE.")
    parser.add_argument("--x_mask_r_thr_ste_tau", type=float, default=0.1,
                        help="STE surrogate temperature for training-time x_mask_r_thr.")
    parser.add_argument("--rec", action="store_true",
                        help="Preserve pre-mask reconstruction channels for x_rec when x-mask is enabled.")

    # token gate config
    parser.add_argument("--x_mask_token_gate_mode", type=str, default="token_all",
                        choices=["static_all", "token_all", "token_deep"])
    parser.add_argument("--no_xw_reorder", action="store_true")
    parser.add_argument("--x_mask_token_gate_deep_ratio", type=float, default=0.5)
    parser.add_argument("--x_mask_token_gate_deep_start", type=int, default=-1)
    parser.add_argument("--x_mask_token_mlp_hidden", type=int, default=0)
    parser.add_argument("--x_mask_token_mlp_chunk_size", type=int, default=1024)
    parser.add_argument("--x_mask_token_mlp_shared", action="store_true")
    parser.add_argument("--x_mask_token_no_mlp_shared", action="store_true")
    parser.add_argument("--x_mask_token_use_layer_scale", action="store_true")
    parser.add_argument("--x_mask_token_no_layer_scale", action="store_true")

    # calibration hyperparams
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--cali_bsz", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate_cost", type=float, default=0.0)
    parser.add_argument("--gate_target", type=float, default=float("nan"))
    parser.add_argument("--token_delta_l2", type=float, default=0.0)
    parser.add_argument("--trainable_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trainable_token_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset_gate_logits", action="store_true")
    parser.add_argument("--deactive_amp", action="store_true")

    # === NEW: joint softmax_alpha params ===
    parser.add_argument("--trainable_alpha", action=argparse.BooleanOptionalAction, default=True,
                        help="Jointly train softmax_alpha per-head with gates.")
    parser.add_argument("--alpha_lr", type=float, default=-1.0,
                        help="Separate LR for alpha params. <0 means use --lr.")
    parser.add_argument("--alpha_reg", type=float, default=0.01,
                        help="Regularization weight: lambda * (alpha - 1)^2")
    parser.add_argument("--teacher_dtype", type=str, default="nvfp4",
                        choices=["nvfp4", "bf16", "mixed"],
                        help="Teacher target for layer-wise distillation: quantized dense nvfp4, raw bf16, or mixed (gate->nvfp4, alpha/output_scale->bf16).")

    # Decoupled training: first train gates (alpha frozen at 1), then alpha (gates frozen)
    parser.add_argument("--decouple_training", action="store_true",
                        help="Two-phase: gate-only first, then alpha-only.")
    parser.add_argument("--gate_epochs", type=int, default=-1,
                        help="Epochs for gate-only phase. -1 = use --epochs.")
    parser.add_argument("--alpha_epochs", type=int, default=-1,
                        help="Epochs for alpha-only phase. -1 = use --epochs.")

    args = parser.parse_args()

    if args.teacher_dtype == "mixed" and not args.decouple_training:
        raise ValueError("--teacher_dtype mixed currently requires --decouple_training so gate and alpha phases can use different teachers.")

    args.exp_name = f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.model_name = args.model.split("/")[-1]
    args.exp_dir = os.path.join(args.output_dir, args.model_name, f"{args.quant_type}", args.exp_name)
    os.makedirs(args.exp_dir, exist_ok=True)
    logger = create_logger(args.exp_dir)
    logger.info('Arguments: ')
    logger.info(pprint.pformat(vars(args)))
    logger.info('--' * 30)

    model_name = _model_name_from_path(args.model)
    dataset_name = args.dataset.lower()

    index_filename = f"./saved/{model_name.lower()}_reorder_index_{dataset_name}_{args.act_sort_metric}.pt"
    select_num_filename = f"./saved/{model_name.lower()}_select_num_{dataset_name}_{args.act_sort_metric}.pt"
    act_scales_filename = f"./saved/{model_name.lower()}_act_scales_{dataset_name}_{args.act_sort_metric}.pt"
    for fn in (index_filename, select_num_filename, act_scales_filename):
        if not os.path.isfile(fn):
            raise FileNotFoundError(f"missing required file: {fn}")

    reorder_index = torch.load(index_filename, weights_only=False)
    select_nums = torch.load(select_num_filename, weights_only=False)

    model, reorder_model_func = _get_model(args.model)
    model.eval()
    model.config.use_cache = False

    teacher_model = None
    teacher_layers = None
    if args.teacher_dtype in {"bf16", "mixed"}:
        teacher_model, _ = _get_model(args.model)
        teacher_model.eval()
        teacher_model.config.use_cache = False
        teacher_layers = teacher_model.model.layers

    x_mask_r_thr = None if args.x_mask_r_thr < 0 else float(args.x_mask_r_thr)

    print("Quantizing model (build x-mask modules)...")
    reorder_kwargs = {
        "device": DEV,
        "kv_cache": False,
        "reorder_index": reorder_index,
        "select_nums": select_nums,
        "quant_type": args.quant_type,
        "reorder_xw": not bool(args.no_xw_reorder),
        "use_x_mask": True,
        "x_mask_tau": float(args.x_mask_tau),
        "x_mask_alpha": float(args.x_mask_alpha),
        "x_mask_skip_layers": args.x_mask_skip_layers,
        "x_mask_r_thr": x_mask_r_thr,
    }
    if "llama" in args.model.lower():
        reorder_kwargs["rec"] = bool(args.rec)
    model = reorder_model_func(model, **reorder_kwargs)

    x_mask_token_mlp_shared = True
    if args.x_mask_token_no_mlp_shared:
        x_mask_token_mlp_shared = False
    elif args.x_mask_token_mlp_shared:
        x_mask_token_mlp_shared = True

    x_mask_token_use_layer_scale = True
    if args.x_mask_token_no_layer_scale:
        x_mask_token_use_layer_scale = False
    elif args.x_mask_token_use_layer_scale:
        x_mask_token_use_layer_scale = True

    configure_x_mask_token_gate(
        model,
        use_x_mask=True,
        x_mask_mode="switch_top2_hard",
        x_mask_token_gate_mode=args.x_mask_token_gate_mode,
        x_mask_token_gate_deep_ratio=float(args.x_mask_token_gate_deep_ratio),
        x_mask_token_gate_deep_start=int(args.x_mask_token_gate_deep_start),
        x_mask_token_mlp_hidden=int(args.x_mask_token_mlp_hidden),
        x_mask_token_mlp_chunk_size=int(args.x_mask_token_mlp_chunk_size),
        x_mask_token_mlp_shared=bool(x_mask_token_mlp_shared),
        x_mask_token_use_layer_scale=bool(x_mask_token_use_layer_scale),
    )

    if x_mask_r_thr is not None:
        for layer in model.model.layers:
            for xm in iter_layer_x_mask_modules(layer):
                xm.x_mask_r_thr = x_mask_r_thr
                xm.x_mask_train_hard_r_thr = bool(args.x_mask_train_hard_r_thr)
                xm.x_mask_r_thr_ste_tau = float(args.x_mask_r_thr_ste_tau)

    print("Catching first-layer inputs...")
    trainloader, _, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=args.seqlen
    )

    layers = model.model.layers
    layers[0] = layers[0].to(DEV)
    model.model.embed_tokens = model.model.embed_tokens.to(DEV)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(DEV)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=DEV)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in trainloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(DEV))
            except ValueError:
                pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None

    fp_inps = inps
    fp_outs = torch.zeros_like(inps)
    teacher_inps = inps.detach().cpu() if args.teacher_dtype in {"bf16", "mixed"} else None
    teacher_outs = torch.zeros_like(teacher_inps) if args.teacher_dtype in {"bf16", "mixed"} else None

    loss_func = torch.nn.MSELoss()
    amp_dtype = torch.float32 if args.deactive_amp else torch.bfloat16

    def _traincast():
        if args.deactive_amp:
            return nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)

    ckpt_layers = {}
    alpha_by_layer = {}
    scale_by_layer = {}
    mlp_scale_by_layer = {}
    alpha_lr = float(args.alpha_lr) if args.alpha_lr > 0 else float(args.lr)

    print("Start JOINT calibration (x_mask gates + softmax_alpha)...")
    for layer_idx in range(len(layers)):
        print(f"========= Layer {layer_idx} =========")
        layer = layers[layer_idx].to(DEV)
        layer.eval()
        teacher_layer = None
        if teacher_layers is not None:
            teacher_layer = teacher_layers[layer_idx].to(DEV)
            teacher_layer.eval()

        # ---- teacher outputs (x_mask disabled, alpha=1) ----
        attn = getattr(layer, "self_attn", None)
        if args.teacher_dtype in {"bf16", "mixed"}:
            with torch.no_grad():
                for start in range(0, args.nsamples, args.cali_bsz):
                    x = teacher_inps[start : start + args.cali_bsz].to(DEV)
                    bs = int(x.shape[0])
                    am = attention_mask_batch[:bs] if attention_mask_batch is not None else None
                    pid = position_ids
                    if pid is not None and pid.shape[0] != bs:
                        pid = pid.repeat(bs, 1)
                    teacher_outs[start : start + bs] = teacher_layer(x, attention_mask=am, position_ids=pid)[0].detach().cpu()

        # nvfp4 teacher: quantized layer with x_mask disabled, alpha=1
        # For "nvfp4" this is the only teacher; for "mixed" this provides
        # the gate-phase target (fp_outs) alongside the bf16 teacher_outs.
        if args.teacher_dtype in {"nvfp4", "mixed"}:
            set_layer_x_mask_alpha(layer, 0.0)
            set_layer_x_mask_eval_mode(layer, False)
            # Reset alpha to 1.0 for teacher
            if attn is not None and hasattr(attn, "softmax_alpha"):
                with torch.no_grad():
                    attn.softmax_alpha.fill_(1.0)

            with torch.no_grad():
                for start in range(0, args.nsamples, args.cali_bsz):
                    x = fp_inps[start : start + args.cali_bsz]
                    bs = int(x.shape[0])
                    am = attention_mask_batch[:bs] if attention_mask_batch is not None else None
                    pid = position_ids
                    if pid is not None and pid.shape[0] != bs:
                        pid = pid.repeat(bs, 1)
                    fp_outs[start : start + bs] = layer(x, attention_mask=am, position_ids=pid)[0]

        # ---- train student: gates + alpha jointly ----
        set_layer_x_mask_alpha(layer, float(args.x_mask_alpha))

        for p in layer.parameters():
            p.requires_grad_(False)

        trainable_gate = []
        trainable_alpha = []

        # Collect x_mask gate params
        for xm in iter_layer_x_mask_modules(layer):
            xm.x_mask_gate_mean_requires_grad = bool(args.gate_cost > 0)
            if args.reset_gate_logits and hasattr(xm, "x_mask_gate_logits"):
                with torch.no_grad():
                    xm.x_mask_gate_logits.data.zero_()

            if args.trainable_gate and hasattr(xm, "x_mask_gate_logits"):
                xm.x_mask_gate_logits.requires_grad_(True)
                trainable_gate.append(xm.x_mask_gate_logits)

            if args.trainable_token_gate and getattr(xm, "x_mask_token_gate_enabled", False):
                mlp = getattr(xm, "x_mask_token_mlp", None)
                if mlp is None and hasattr(xm, "_ensure_x_mask_token_mlp"):
                    mlp = xm._ensure_x_mask_token_mlp()
                if mlp is not None:
                    for p in mlp.parameters():
                        p.requires_grad_(True)
                        trainable_gate.append(p)
                if x_mask_token_use_layer_scale:
                    scale = getattr(xm, "x_mask_token_scale", None)
                    if scale is not None:
                        scale.requires_grad_(True)
                        trainable_gate.append(scale)

        # Collect softmax_alpha + output_scale params
        if args.trainable_alpha and attn is not None:
            if hasattr(attn, "softmax_alpha"):
                attn.softmax_alpha.requires_grad_(True)
                trainable_alpha.append(attn.softmax_alpha)
            if hasattr(attn, "output_scale"):
                attn.output_scale.requires_grad_(True)
                trainable_alpha.append(attn.output_scale)
        mlp = getattr(layer, "mlp", None)
        if args.trainable_alpha and mlp is not None and hasattr(mlp, "mlp_output_scale"):
            mlp.mlp_output_scale.requires_grad_(True)
            trainable_alpha.append(mlp.mlp_output_scale)

        trainable_gate = _unique_params(trainable_gate)
        trainable_alpha = _unique_params(trainable_alpha)

        # Use separate param groups for gate and alpha (allow different LR)
        # Build phase schedule: [(phase_name, optimizer, n_epochs), ...]
        if args.decouple_training:
            gate_n = int(args.gate_epochs) if args.gate_epochs >= 0 else int(args.epochs)
            alpha_n = int(args.alpha_epochs) if args.alpha_epochs >= 0 else int(args.epochs)
            phase_schedule = []
            if trainable_gate and gate_n > 0:
                phase_schedule.append((
                    "gate-only",
                    torch.optim.AdamW([{"params": trainable_gate, "lr": float(args.lr)}]),
                    gate_n,
                ))
            if trainable_alpha and alpha_n > 0:
                phase_schedule.append((
                    "alpha-only",
                    torch.optim.AdamW([{"params": trainable_alpha, "lr": alpha_lr}]),
                    alpha_n,
                ))
        else:
            param_groups = []
            if trainable_gate:
                param_groups.append({"params": trainable_gate, "lr": float(args.lr)})
            if trainable_alpha:
                param_groups.append({"params": trainable_alpha, "lr": alpha_lr})
            _joint_opt = torch.optim.AdamW(param_groups) if param_groups else None
            phase_schedule = [("joint", _joint_opt, int(args.epochs))] if _joint_opt else []

        for phase_name, phase_opt, phase_epochs in phase_schedule:
            if args.decouple_training:
                if phase_name == "gate-only":
                    for p in trainable_gate:
                        p.requires_grad_(True)
                    for p in trainable_alpha:
                        p.requires_grad_(False)
                    if attn is not None and hasattr(attn, "softmax_alpha"):
                        attn.softmax_alpha.requires_grad_(False)
                else:  # alpha-only
                    for p in trainable_gate:
                        p.requires_grad_(False)
                    for p in trainable_alpha:
                        p.requires_grad_(True)
                    if attn is not None and hasattr(attn, "softmax_alpha"):
                        attn.softmax_alpha.requires_grad_(True)

            for epoch in range(phase_epochs):
                mse = 0
                for start in range(0, args.nsamples, args.cali_bsz):
                    x = fp_inps[start : start + args.cali_bsz]
                    y_ref = teacher_outs[start : start + args.cali_bsz].to(DEV) if teacher_outs is not None else fp_outs[start : start + args.cali_bsz]
                    y_ref_nvfp4 = fp_outs[start : start + args.cali_bsz]
                    bs = int(x.shape[0])
                    am = attention_mask_batch[:bs] if attention_mask_batch is not None else None
                    pid = position_ids
                    if pid is not None and pid.shape[0] != bs:
                        pid = pid.repeat(bs, 1)

                    with _traincast():
                        y = layer(x, attention_mask=am, position_ids=pid)[0]
                        if args.teacher_dtype == "mixed":
                            loss_gate = loss_func(y, y_ref_nvfp4)
                            loss_alpha = loss_func(y, y_ref)
                            if phase_name == "gate-only":
                                loss = loss_gate
                            elif phase_name == "alpha-only":
                                loss = loss_alpha
                            else:
                                loss = loss_alpha
                        else:
                            loss = loss_func(y, y_ref)

                        # Gate sparsity cost
                        if args.gate_cost > 0:
                            gate_cost = None
                            target = None if math.isnan(float(args.gate_target)) else float(args.gate_target)
                            for xm in iter_layer_x_mask_modules(layer):
                                gm = getattr(xm, "_last_x_mask_gate_mean_grad", None)
                                if gm is None:
                                    continue
                                cost = gm if target is None else (gm - target) ** 2
                                gate_cost = cost if gate_cost is None else gate_cost + cost
                            if gate_cost is not None:
                                loss = loss + float(args.gate_cost) * gate_cost

                        # Token delta L2
                        if args.token_delta_l2 > 0:
                            delta_l2 = None
                            for xm in iter_layer_x_mask_modules(layer):
                                d = getattr(xm, "_last_x_mask_gate_delta_l2", None)
                                if d is not None:
                                    delta_l2 = d if delta_l2 is None else delta_l2 + d
                            if delta_l2 is not None:
                                loss = loss + float(args.token_delta_l2) * delta_l2

                        # Alpha + output_scale regularization (skip during gate-only phase)
                        if args.alpha_reg > 0 and trainable_alpha and phase_name != "gate-only":
                            if attn is not None and hasattr(attn, "softmax_alpha"):
                                alpha_dev = attn.softmax_alpha.float() - 1.0
                                loss = loss + args.alpha_reg * (alpha_dev * alpha_dev).mean()
                            if attn is not None and hasattr(attn, "output_scale"):
                                scale_dev = attn.output_scale.float() - 1.0
                                loss = loss + args.alpha_reg * (scale_dev * scale_dev).mean()
                            if mlp is not None and hasattr(mlp, "mlp_output_scale"):
                                mlp_scale_dev = mlp.mlp_output_scale.float() - 1.0
                                loss = loss + args.alpha_reg * (mlp_scale_dev * mlp_scale_dev).mean()

                    mse += loss.detach().cpu()
                    loss = loss / loss.clone().detach().clamp_min(1e-12)
                    phase_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    phase_opt.step()

                cur_lr = phase_opt.param_groups[0]["lr"] if len(phase_opt.param_groups) > 0 else float("nan")
                logger.info(f"layer {layer_idx} [{phase_name}] iter {epoch}, lr {cur_lr:.8f}, mse: {mse:.8f}")

                # Log gate stats
                stats_parts = []
                for name, mask in (
                    ("self_attn.x_mask_in", getattr(layer.self_attn, "x_mask_in", None)),
                    ("self_attn.x_mask_out", getattr(layer.self_attn, "x_mask_out", None)),
                    ("mlp.x_mask_up", getattr(layer.mlp, "x_mask_up", None)),
                    ("mlp.x_mask_down", getattr(layer.mlp, "x_mask_down", None)),
                ):
                    if mask is None or not getattr(mask, "use_x_mask", False):
                        continue
                    mean = getattr(mask, "_last_x_mask_gate_mean", None)
                    if mean is None:
                        continue
                    std = getattr(mask, "_last_x_mask_gate_std", None)
                    frac_low = getattr(mask, "_last_x_mask_gate_frac_low", None)
                    frac_high = getattr(mask, "_last_x_mask_gate_frac_high", None)
                    stats_parts.append(
                        f"{name}: mean={float(mean):.3f} std={float(std) if std is not None else float('nan'):.3f} "
                        f"low={float(frac_low) if frac_low is not None else float('nan'):.3f} "
                        f"high={float(frac_high) if frac_high is not None else float('nan'):.3f}"
                    )
                if stats_parts:
                    logger.info("gate_stats: " + " | ".join(stats_parts))

                # Log alpha + output_scale stats (skipped during gate-only phase)
                if phase_name != "gate-only" and attn is not None:
                    if hasattr(attn, "softmax_alpha"):
                        a = attn.softmax_alpha.detach().float()
                        logger.info(
                            f"softmax_alpha: mean={a.mean():.4f} min={a.min():.4f} max={a.max():.4f} "
                            f"std={a.std():.4f} |a-1|_mean={((a-1).abs()).mean():.4f}"
                        )
                    if hasattr(attn, "output_scale"):
                        s = attn.output_scale.detach().float()
                        logger.info(
                            f"output_scale: mean={s.mean():.4f} min={s.min():.4f} max={s.max():.4f} "
                            f"std={s.std():.4f} |s-1|_mean={(s-1).abs().mean():.4f}"
                        )
                    if mlp is not None and hasattr(mlp, "mlp_output_scale"):
                        ms = mlp.mlp_output_scale.detach().float()
                        logger.info(
                            f"mlp_output_scale: mean={ms.mean():.4f} min={ms.min():.4f} max={ms.max():.4f} "
                            f"std={ms.std():.4f} |s-1|_mean={(ms-1).abs().mean():.4f}"
                        )

        # ---- cleanup optimizer and cached graph refs ----
        for _, _opt, _ in phase_schedule:
            if _opt is not None:
                del _opt
        del phase_schedule, trainable_gate, trainable_alpha
        for xm in iter_layer_x_mask_modules(layer):
            for attr in list(vars(xm)):
                if attr.startswith("_last_x_mask"):
                    setattr(xm, attr, None)

        # ---- save layer state ----
        layer_xmask_state = {
            k: v.detach().cpu() for k, v in layer.state_dict().items() if "x_mask" in k
        }
        ckpt_layers[layer_idx] = layer_xmask_state

        # Save alpha + output_scale for this layer, freeze both
        if attn is not None and hasattr(attn, "softmax_alpha"):
            alpha_by_layer[layer_idx] = attn.softmax_alpha.detach().cpu().clone()
            attn.softmax_alpha.requires_grad_(False)
        if attn is not None and hasattr(attn, "output_scale"):
            scale_by_layer[layer_idx] = attn.output_scale.detach().cpu().clone()
            attn.output_scale.requires_grad_(False)
        if mlp is not None and hasattr(mlp, "mlp_output_scale"):
            mlp_scale_by_layer[layer_idx] = mlp.mlp_output_scale.detach().cpu().clone()
            mlp.mlp_output_scale.requires_grad_(False)

        # ---- next layer inputs ----
        # Use the TRAINED student output (with x_mask + alpha) as input to next layer
        set_layer_x_mask_eval_mode(layer, False)
        with torch.no_grad():
            for start in range(0, args.nsamples, args.cali_bsz):
                x = fp_inps[start : start + args.cali_bsz]
                bs = int(x.shape[0])
                am = attention_mask_batch[:bs] if attention_mask_batch is not None else None
                pid = position_ids
                if pid is not None and pid.shape[0] != bs:
                    pid = pid.repeat(bs, 1)
                fp_outs[start : start + bs] = layer(x, attention_mask=am, position_ids=pid)[0]

        fp_inps, fp_outs = fp_outs, fp_inps
        if teacher_outs is not None:
            teacher_inps, teacher_outs = teacher_outs, teacher_inps

        layers[layer_idx] = layer.cpu()
        if teacher_layer is not None:
            teacher_layers[layer_idx] = teacher_layer.cpu()
        del layer
        if teacher_layer is not None:
            del teacher_layer
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Build combined alpha + output_scale tensors ----
    n_layers = len(layers)
    num_heads = None
    for v in alpha_by_layer.values():
        num_heads = v.numel()
        break
    if num_heads is None:
        num_heads = 1
    softmax_alpha = torch.ones((n_layers, num_heads), dtype=torch.float32)
    for i, a in alpha_by_layer.items():
        softmax_alpha[i] = a.view(-1)

    hidden_size_ckpt = None
    for v in scale_by_layer.values():
        hidden_size_ckpt = v.numel()
        break
    if hidden_size_ckpt is None:
        hidden_size_ckpt = 1
    output_scale = torch.ones((n_layers, hidden_size_ckpt), dtype=torch.float32)
    for i, s in scale_by_layer.items():
        output_scale[i] = s.view(-1)

    hidden_size_mlp_ckpt = None
    for v in mlp_scale_by_layer.values():
        hidden_size_mlp_ckpt = v.numel()
        break
    if hidden_size_mlp_ckpt is None:
        hidden_size_mlp_ckpt = 1
    mlp_output_scale = torch.ones((n_layers, hidden_size_mlp_ckpt), dtype=torch.float32)
    for i, s in mlp_scale_by_layer.items():
        mlp_output_scale[i] = s.view(-1)

    # ---- Save checkpoint ----
    out_path = os.path.join(args.exp_dir, f"{model_name.lower()}_joint_{dataset_name}_{args.act_sort_metric}_{args.quant_type}.pt")
    torch.save(
        {
            "meta": {
                "model": args.model,
                "dataset": args.dataset,
                "act_sort_metric": args.act_sort_metric,
                "quant_type": args.quant_type,
                "x_mask_tau": float(args.x_mask_tau),
                "x_mask_alpha": float(args.x_mask_alpha),
                "x_mask_r_thr": x_mask_r_thr,
                "x_mask_train_hard_r_thr": bool(args.x_mask_train_hard_r_thr),
                "x_mask_r_thr_ste_tau": float(args.x_mask_r_thr_ste_tau),
                "x_mask_token_gate_mode": args.x_mask_token_gate_mode,
                "x_mask_token_gate_deep_ratio": float(args.x_mask_token_gate_deep_ratio),
                "x_mask_token_gate_deep_start": int(args.x_mask_token_gate_deep_start),
                "x_mask_token_mlp_hidden": int(args.x_mask_token_mlp_hidden),
                "x_mask_token_mlp_chunk_size": int(args.x_mask_token_mlp_chunk_size),
                "x_mask_token_mlp_shared": bool(x_mask_token_mlp_shared),
                "x_mask_token_use_layer_scale": bool(x_mask_token_use_layer_scale),
                "trainable_alpha": bool(args.trainable_alpha),
                "alpha_lr": alpha_lr,
                "alpha_reg": float(args.alpha_reg),
                "teacher_dtype": args.teacher_dtype,
                "joint": True,
            },
            "layers": ckpt_layers,
            "softmax_alpha": softmax_alpha,
            "output_scale": output_scale,
            "mlp_output_scale": mlp_output_scale,
        },
        out_path,
    )
    print(f"Saved joint checkpoint: {out_path}")
    print(f"softmax_alpha:  mean={softmax_alpha.mean():.4f} min={softmax_alpha.min():.4f} max={softmax_alpha.max():.4f}")
    print(f"output_scale:   mean={output_scale.mean():.4f} min={output_scale.min():.4f} max={output_scale.max():.4f}")
    print(f"mlp_output_scale: mean={mlp_output_scale.mean():.4f} min={mlp_output_scale.min():.4f} max={mlp_output_scale.max():.4f}")


if __name__ == "__main__":
    main()
