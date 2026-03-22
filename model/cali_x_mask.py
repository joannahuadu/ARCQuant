import argparse
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
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "pile", "humaneval"],
    )
    parser.add_argument(
        "--act_sort_metric",
        type=str,
        default="max",
        choices=["mean", "frobenius", "hessian", "max"],
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="NVFP4",
        choices=["NVFP4", "MXFP4", "INT4", "HiF4"],
    )
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory path.")
    parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name.")
    # x-mask config
    parser.add_argument("--x_mask_tau", type=float, default=1.0)
    parser.add_argument("--x_mask_alpha", type=float, default=1.0)
    parser.add_argument("--x_mask_skip_layers", type=str, default="", help="Comma/range list of layer ids to skip x-mask, e.g. '0,1,8-15'.")
    parser.add_argument("--x_mask_r_thr", type=float, default=-1.0)

    # token gate config (FlatQuant compatible)
    parser.add_argument(
        "--x_mask_token_gate_mode",
        type=str,
        default="token_all",
        choices=["static_all", "token_all", "token_deep"],
    )
    parser.add_argument(
        "--no_xw_reorder",
        action="store_true",
        help="Disable channel reordering for both activations (X) and weights (W).",
    )
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

    args = parser.parse_args()
    
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

    x_mask_r_thr = None if args.x_mask_r_thr < 0 else float(args.x_mask_r_thr)

    print("Quantizing model (build x-mask modules)...")
    model = reorder_model_func(
        model,
        device=DEV,
        kv_cache=False,
        reorder_index=reorder_index,
        select_nums=select_nums,
        quant_type=args.quant_type,
        reorder_xw=not bool(args.no_xw_reorder),
        use_x_mask=True,
        x_mask_tau=float(args.x_mask_tau),
        x_mask_alpha=float(args.x_mask_alpha),
        x_mask_skip_layers=args.x_mask_skip_layers,
        x_mask_r_thr=x_mask_r_thr,
    )

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
                sample = batch[0]
                model(sample.to(DEV))
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

    loss_func = torch.nn.MSELoss()
    amp_dtype = torch.float32 if args.deactive_amp else torch.bfloat16

    def _traincast():
        if args.deactive_amp:
            return nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)

    ckpt_layers = {}
    print("Start x-mask calibration (teacher = x_mask disabled)...")
    for layer_idx in range(len(layers)):
        print(f"========= Layer {layer_idx} =========")
        layer = layers[layer_idx].to(DEV)
        layer.eval()

        # ---- teacher outputs (x_mask disabled) ----
        set_layer_x_mask_alpha(layer, 0.0)
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

        # ---- train student gates ----
        set_layer_x_mask_alpha(layer, float(args.x_mask_alpha))

        for p in layer.parameters():
            p.requires_grad_(False)

        trainable = []
        for xm in iter_layer_x_mask_modules(layer):
            xm.x_mask_gate_mean_requires_grad = bool(args.gate_cost > 0)
            if args.reset_gate_logits and hasattr(xm, "x_mask_gate_logits"):
                with torch.no_grad():
                    xm.x_mask_gate_logits.data.zero_()

            if args.trainable_gate and hasattr(xm, "x_mask_gate_logits"):
                xm.x_mask_gate_logits.requires_grad_(True)
                trainable.append(xm.x_mask_gate_logits)

            if args.trainable_token_gate and getattr(xm, "x_mask_token_gate_enabled", False):
                mlp = getattr(xm, "x_mask_token_mlp", None)
                if mlp is None and hasattr(xm, "_ensure_x_mask_token_mlp"):
                    mlp = xm._ensure_x_mask_token_mlp()
                if mlp is not None:
                    for p in mlp.parameters():
                        p.requires_grad_(True)
                        trainable.append(p)
                if x_mask_token_use_layer_scale:
                    scale = getattr(xm, "x_mask_token_scale", None)
                    if scale is not None:
                        scale.requires_grad_(True)
                        trainable.append(scale)

        trainable = _unique_params(trainable)
        optimizer = torch.optim.AdamW(trainable, lr=float(args.lr)) if trainable else None

        if optimizer is not None:
            for epoch in range(int(args.epochs)):
                mse = 0
                for start in range(0, args.nsamples, args.cali_bsz):
                    x = fp_inps[start : start + args.cali_bsz]
                    y_ref = fp_outs[start : start + args.cali_bsz]
                    bs = int(x.shape[0])
                    am = attention_mask_batch[:bs] if attention_mask_batch is not None else None
                    pid = position_ids
                    if pid is not None and pid.shape[0] != bs:
                        pid = pid.repeat(bs, 1)

                    with _traincast():
                        y = layer(x, attention_mask=am, position_ids=pid)[0]
                        loss = loss_func(y, y_ref)

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

                        if args.token_delta_l2 > 0:
                            delta_l2 = None
                            for xm in iter_layer_x_mask_modules(layer):
                                d = getattr(xm, "_last_x_mask_gate_delta_l2", None)
                                if d is not None:
                                    delta_l2 = d if delta_l2 is None else delta_l2 + d
                            if delta_l2 is not None:
                                loss = loss + float(args.token_delta_l2) * delta_l2
                    mse += loss.detach().cpu()
                    loss = loss / loss.clone().detach().clamp_min(1e-12)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                
                cur_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else float("nan")
                logger.info(f"layer {layer_idx} lwc lac iter {epoch}, lr {cur_lr:.8f}, mse: {mse:.8f}" )
                if (
                    (getattr(args, "trainable_gate", False) or getattr(args, "trainable_token_gate", False))
                ):
                    stats_parts = []
                    for name, mask in (
                        ("self_attn.x_mask_in", layer.self_attn.x_mask_in),
                        ("self_attn.x_mask_out", layer.self_attn.x_mask_out),
                        ("mlp.x_mask_up", layer.mlp.x_mask_up),
                        ("mlp.x_mask_down", layer.mlp.x_mask_down),
                    ):
                        if mask is None or not getattr(mask, "use_x_mask", False):
                            continue
                        mean = getattr(mask, "_last_x_mask_gate_mean", None)
                        if mean is None:
                            continue
                        std = getattr(mask, "_last_x_mask_gate_std", None)
                        frac_low = getattr(mask, "_last_x_mask_gate_frac_low", None)
                        frac_high = getattr(mask, "_last_x_mask_gate_frac_high", None)
                        tok_var = getattr(mask, "_last_x_mask_gate_tok_var", None)
                        delta_l2 = getattr(mask, "_last_x_mask_gate_delta_l2", None)
                        stats_parts.append(
                            f"{name}: mean={float(mean):.3f} std={float(std) if std is not None else float('nan'):.3f} "
                            f"low={float(frac_low) if frac_low is not None else float('nan'):.3f} "
                            f"high={float(frac_high) if frac_high is not None else float('nan'):.3f} "
                            f"tok_var={float(tok_var) if tok_var is not None else float('nan'):.3e} "
                            f"delta_l2={float(delta_l2) if delta_l2 is not None else float('nan'):.3e}"
                        )
                    if stats_parts:
                        logger.info("x_mask_gate_stats: " + " | ".join(stats_parts))

        # ---- save layer x-mask params ----
        layer_xmask_state = {
            k: v.detach().cpu() for k, v in layer.state_dict().items() if "x_mask" in k
        }
        ckpt_layers[layer_idx] = layer_xmask_state

        # ---- next layer inputs (teacher outputs) ----
        fp_inps, fp_outs = fp_outs, fp_inps

        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    out_path = os.path.join(args.exp_dir, f"{model_name.lower()}_xmask_{dataset_name}_{args.act_sort_metric}_{args.quant_type}.pt")
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
                "x_mask_token_gate_mode": args.x_mask_token_gate_mode,
                "x_mask_token_gate_deep_ratio": float(args.x_mask_token_gate_deep_ratio),
                "x_mask_token_gate_deep_start": int(args.x_mask_token_gate_deep_start),
                "x_mask_token_mlp_hidden": int(args.x_mask_token_mlp_hidden),
                "x_mask_token_mlp_chunk_size": int(args.x_mask_token_mlp_chunk_size),
                "x_mask_token_mlp_shared": bool(x_mask_token_mlp_shared),
                "x_mask_token_use_layer_scale": bool(x_mask_token_use_layer_scale),
            },
            "layers": ckpt_layers,
        },
        out_path,
    )
    print(f"Saved x-mask checkpoint: {out_path}")


if __name__ == "__main__":
    main()
