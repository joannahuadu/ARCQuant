"""Evaluate bf16 hook checkpoints produced by cali_x_mask_bf16/cali_joint_bf16."""
import argparse
import json
import random
from pathlib import Path

import lm_eval
import numpy as np
import torch
from lm_eval import utils as lm_eval_utils
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

from bf16_hook_utils import apply_bf16_joint_plus
from datautils import get_loaders
from eval import eval_ppl
from x_mask_utils import configure_x_mask_token_gate, iter_layer_x_mask_modules, set_layer_x_mask_eval_mode


def _first_real_layer_device(layer) -> torch.device:
    for name, param in layer.named_parameters():
        if "x_mask" in name or "output_low_rank" in name or name.endswith(("softmax_alpha", "output_scale", "mlp_output_scale")):
            continue
        if param.device.type != "meta":
            return param.device
    for name, buffer in layer.named_buffers():
        if "x_mask" in name or "output_low_rank" in name or name.endswith(("softmax_alpha", "output_scale", "mlp_output_scale")):
            continue
        if buffer.device.type != "meta":
            return buffer.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _move_joint_modules_to_layer_device(layer) -> None:
    device = _first_real_layer_device(layer)
    for xm in iter_layer_x_mask_modules(layer):
        xm.to(device=device)

    attn = getattr(layer, "self_attn", None)
    if attn is not None:
        for name in ("softmax_alpha", "output_scale"):
            tensor = getattr(attn, name, None)
            if tensor is not None:
                setattr(attn, name, tensor.to(device=device))
        low_rank = getattr(attn, "output_low_rank", None)
        if low_rank is not None:
            attn.output_low_rank = low_rank.to(device=device)

    mlp = getattr(layer, "mlp", None)
    tensor = getattr(mlp, "mlp_output_scale", None) if mlp is not None else None
    if tensor is not None:
        setattr(mlp, "mlp_output_scale", tensor.to(device=device))
    low_rank = getattr(mlp, "output_low_rank", None) if mlp is not None else None
    if low_rank is not None:
        mlp.output_low_rank = low_rank.to(device=device)


def _load_bf16_hook_checkpoint(model, ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported bf16 checkpoint format: {ckpt_path}")

    meta = ckpt.get("meta", {}) or {}
    x_mask_r_thr = meta.get("x_mask_r_thr", -1.0)
    x_mask_r_thr = None if x_mask_r_thr is None or float(x_mask_r_thr) < 0 else float(x_mask_r_thr)
    print(x_mask_r_thr)
    use_attn_output_scale = bool(meta.get("train_attn_output_scale", False)) or "output_scale" in ckpt
    use_mlp_output_scale = bool(meta.get("train_mlp_output_scale", False)) or "mlp_output_scale" in ckpt

    model = apply_bf16_joint_plus(
        model,
        use_x_mask=True,
        x_mask_tau=float(meta.get("x_mask_tau", 1.0)),
        x_mask_alpha=float(meta.get("x_mask_alpha", 1.0)),
        x_mask_r_thr=x_mask_r_thr,
        token_gate_mode=meta.get("x_mask_token_gate_mode", "token_all"),
        token_mlp_hidden=int(meta.get("x_mask_token_mlp_hidden", 0)),
        token_mlp_chunk_size=int(meta.get("x_mask_token_mlp_chunk_size", 1024)),
        token_use_layer_scale=bool(meta.get("x_mask_token_use_layer_scale", True)),
        use_attn_output_scale=use_attn_output_scale,
        use_mlp_output_scale=use_mlp_output_scale,
        attn_low_rank_layers=meta.get("attn_low_rank_layers", ""),
        attn_low_rank_rank=int(meta.get("attn_low_rank_rank", 0) or 0),
        mlp_low_rank_layers=meta.get("mlp_low_rank_layers", ""),
        mlp_low_rank_rank=int(meta.get("mlp_low_rank_rank", 0) or 0),
    )
    configure_x_mask_token_gate(
        model,
        use_x_mask=True,
        x_mask_mode="switch_top2_hard",
        x_mask_token_gate_mode=meta.get("x_mask_token_gate_mode", "token_all"),
        x_mask_token_gate_deep_ratio=float(meta.get("x_mask_token_gate_deep_ratio", 0.5)),
        x_mask_token_gate_deep_start=int(meta.get("x_mask_token_gate_deep_start", -1)),
        x_mask_token_mlp_hidden=int(meta.get("x_mask_token_mlp_hidden", 0)),
        x_mask_token_mlp_chunk_size=int(meta.get("x_mask_token_mlp_chunk_size", 1024)),
        x_mask_token_mlp_shared=bool(meta.get("x_mask_token_mlp_shared", True)),
        x_mask_token_use_layer_scale=bool(meta.get("x_mask_token_use_layer_scale", True)),
    )

    layers = getattr(model.model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose model.layers")
    for key, state in (ckpt.get("layers", {}) or {}).items():
        idx = int(key)
        if idx < 0 or idx >= len(layers):
            continue
        layers[idx].load_state_dict(state, strict=False, assign=True)
        _move_joint_modules_to_layer_device(layers[idx])
        set_layer_x_mask_eval_mode(layers[idx], True)

    # Ensure all x-mask modules run in eval mode during benchmarking, even for
    # layers that do not receive checkpoint state.
    for layer in layers:
        set_layer_x_mask_eval_mode(layer, True)

    meta = dict(meta)
    meta.update(
        {
            "path": ckpt_path,
            "use_attn_output_scale": use_attn_output_scale,
            "use_mlp_output_scale": use_mlp_output_scale,
        }
    )
    return meta


def _task_names(tasks):
    task_aliases = {"ceval": "ceval-valid"}
    task_patterns = []
    for item in tasks:
        task_patterns.extend([x.strip() for x in str(item).split(",") if x.strip()])
    task_patterns = [task_aliases.get(t, t) for t in task_patterns]
    task_manager = TaskManager()
    return sorted(set(lm_eval_utils.pattern_match(task_patterns, task_manager.all_tasks)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--bf16_ckpt", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", default=["arc_challenge"])
    parser.add_argument("--lm_eval_batch_size", "--batch_size", type=int, default=4)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size_overrides", type=str, default='{"mmlu": 1, "ceval-valid": 1}')
    parser.add_argument("--fewshot_overrides", type=str, default='{"mmlu": 5, "ceval-valid": 5}')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    print("Loading bf16 model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    model.config.use_cache = False

    meta = _load_bf16_hook_checkpoint(model, args.bf16_ckpt)
    print(f"Loaded bf16 hook ckpt meta: {meta}")

    if args.eval_ppl:
        _, testloader, _ = get_loaders("wikitext2", seed=args.seed, model=args.model, seqlen=2048)
        print("Evaluating wikitext2 perplexity...")
        ppl = eval_ppl(model, testloader)
        print(f"Result,wikitext2,{ppl:.3f}")

    fewshot_overrides = json.loads(args.fewshot_overrides) if args.fewshot_overrides else {}
    batch_size_overrides = json.loads(args.batch_size_overrides) if args.batch_size_overrides else {}
    results_by_task = {}
    for task_name in _task_names(args.tasks):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model.eval()
        task_bsz = next(
            (
                v
                for k, v in batch_size_overrides.items()
                if task_name == k or task_name.startswith(k + "_")
            ),
            args.lm_eval_batch_size,
        )
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=task_bsz)
        task_fewshot = fewshot_overrides.get(task_name, args.num_fewshot)
        print(f"\nEvaluating {task_name} ({task_fewshot}-shot, batch_size={task_bsz})...")
        result = lm_eval.simple_evaluate(
            hflm,
            tasks=[task_name],
            num_fewshot=task_fewshot,
            batch_size=task_bsz,
        )
        results_by_task[task_name] = result.get("results", {}).get(task_name, {})

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for task, metrics in results_by_task.items():
        for key, value in metrics.items():
            if "stderr" not in key and "acc" in key.lower():
                print(f"{task} {key}: {value}")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results_by_task, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
