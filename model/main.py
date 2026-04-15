import torch
from collections import defaultdict

from model_utils import reorder_model_llama, reorder_model_qwen, reorder_model_mixtral
from parallel_utils import map_layers_to_multi_gpus
from datautils import get_loaders
from eval import *

import time

from visualize import *
from x_mask_utils import (
    iter_layer_x_mask_modules,
    load_x_mask_checkpoint,
    parse_layer_spec,
    set_layer_x_mask_alpha,
    set_layer_x_mask_eval_mode,
)
from softmax_alpha_utils import load_softmax_alpha_checkpoint


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    # model.seqlen = 2048
    return model

def get_qwen(model):
    import torch
    def skip(*args, **kwargs):
        pass
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
   
    return model

def get_mixtral(model):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
   
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, 
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--act_sort_metric', type=str, default='max', choices=['mean', 'frobenius', 'hessian', 'max'],
        help='The metric used to sort the activations.'
    )
   
    parser.add_argument(
        '--kv_cache', action='store_true',
        help='Whether to quant KV_Cache'
    )

    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"],
        help='Tasks to evaluate on LM Eval.')
    parser.add_argument(
        "--eval_ppl", action="store_true",
        help='Whether to evaluate perplexity.'
    )
    parser.add_argument('--output_file', type=str, default=None, help='Optional path to save lm_eval results as JSON.')
    parser.add_argument('--lm_eval_batch_size', type=int, default=16, help='Batch size for evaluation with lm eval harness.')
    parser.add_argument('--num_fewshot', type=int, default=0, help='Number of few-shot examples for lm_eval.')
    parser.add_argument('--batch_size_overrides', type=str, default='{"mmlu": 1, "ceval-valid": 1}',
        help='JSON dict of per-task batch size overrides to prevent OOM on long 5-shot contexts.')
    parser.add_argument('--fewshot_overrides', type=str, default='{"mmlu": 5, "ceval-valid": 5}',
        help='JSON dict of per-task fewshot overrides, e.g. \'{"mmlu": 5, "ceval-valid": 5}\'')
    parser.add_argument(
        "--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "pile", "humaneval", "arc_mix", "wikitext2_c4_mix_1to1", "wikitext2_c4_mix_3to1"], 
        help="The calibration dataset to use."
    )
    parser.add_argument(
        "--quant_type", type=str, default="NVFP4", choices=["NVFP4", "MXFP4", "INT4", "HiF4"], 
        help="data type for W and A quantization."
    )
    parser.add_argument(
        "--no_xw_reorder",
        action="store_true",
        help="Disable channel reordering for both activations (X) and weights (W).",
    )

    # ---- activation 2:4 x-mask (FlatQuant-style switch_top2_hard) ----
    parser.add_argument(
        "--use_x_mask",
        action="store_true",
        help="Enable 2:4 x-mask (switch_top2_hard) with learnable per-token gates.",
    )
    parser.add_argument(
        "--x_mask_ckpt",
        type=str,
        default=None,
        help="Load x-mask checkpoint produced by `python model/cali_x_mask.py ...`.",
    )
    parser.add_argument("--x_mask_tau", type=float, default=1.0)
    parser.add_argument("--x_mask_alpha", type=float, default=1.0)
    parser.add_argument("--x_mask_skip_layers", type=str, default="", help="Comma/range list of layer ids to skip x-mask, e.g. '0,1,8-15'.")
    parser.add_argument(
        "--x_mask_r_thr",
        type=float,
        default=-1.0,
        help="Enable hard switch at eval: r<thr -> apply 2:4 mask. Set <0 to disable.",
    )
    parser.add_argument(
        "--x_mask_eval_hard",
        action="store_true",
        help="Set x-mask modules to eval-hard mode (enforce 2:4 sparsity when r<thr).",
    )
    parser.add_argument(
        "--rec",
        action="store_true",
        help="Preserve pre-mask reconstruction channels for x_rec when x-mask is enabled.",
    )
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
    parser.add_argument("--attn_low_rank_layers", type=str, default="", help="Comma/range list of attention layers that enable output low-rank residual.")
    parser.add_argument("--attn_low_rank_rank", type=int, default=0, help="Low-rank rank for selected attention layers.")
    parser.add_argument("--mlp_low_rank_layers", type=str, default="", help="Comma/range list of MLP layers that enable output low-rank residual.")
    parser.add_argument("--mlp_low_rank_rank", type=int, default=0, help="Low-rank rank for selected MLP layers.")
  
    
    args = parser.parse_args()

    model_name = args.model.split('/')[-2] if len(args.model.split('/')[-1]) == 0 else args.model.split('/')[-1]
    assert model_name != None, "Please check the model path."

    if "llama" in args.model.lower():
        model = get_llama(args.model)
        reorder_model_func = reorder_model_llama
       
    elif "qwen" in args.model.lower():
        model = get_qwen(args.model)
        reorder_model_func = reorder_model_qwen
    
    elif "mixtral" in args.model.lower():
        model = get_mixtral(args.model)
        reorder_model_func = reorder_model_mixtral
       
    model.eval()

    import os

    dataset_name = args.dataset.lower()
    index_filename = f'./saved/{model_name.lower()}_reorder_index_{dataset_name}_{args.act_sort_metric}.pt'
    select_num_filename = f'./saved/{model_name.lower()}_select_num_{dataset_name}_{args.act_sort_metric}.pt'
    act_scales_filename = f'./saved/{model_name.lower()}_act_scales_{dataset_name}_{args.act_sort_metric}.pt'
 
    
    assert os.path.isfile(index_filename), "reorder index file not found."

    print("Loading cached reording index from disk...")
    reorder_index = torch.load(index_filename, weights_only=False)
    select_nums = torch.load(select_num_filename, weights_only=False)
    act_scales = torch.load(act_scales_filename, weights_only=False)

    
    torch.cuda.reset_max_memory_allocated()
    print("Reordering model...")
    start_time=time.time()
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
        "attn_low_rank_layers": args.attn_low_rank_layers,
        "attn_low_rank_rank": int(args.attn_low_rank_rank),
        "mlp_low_rank_layers": args.mlp_low_rank_layers,
        "mlp_low_rank_rank": int(args.mlp_low_rank_rank),
    }
    if "llama" in args.model.lower():
        reorder_kwargs["rec"] = bool(args.rec)
    model = reorder_model_func(
        model,
        **reorder_kwargs,
    )
    model.eval()
    end_time=time.time()
    peak_memory = torch.cuda.max_memory_allocated()

    if args.use_x_mask and args.x_mask_ckpt:
        meta = load_x_mask_checkpoint(model, args.x_mask_ckpt)
        if meta:
            print(f"Loaded x-mask ckpt meta: {meta}")

    if args.softmax_alpha_ckpt:
        meta = load_softmax_alpha_checkpoint(
            model,
            args.softmax_alpha_ckpt,
            skip_layers=args.softmax_alpha_skip_layers,
        )
        if meta:
            print(f"Loaded softmax alpha ckpt meta: {meta}")

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
                set_layer_x_mask_eval_mode(layer, True)


    print(model)
    print(f"Quantized Model Size: {peak_memory/(1024*1024*1024):.2f} GB")
    print(f"Quantized Type is: {args.quant_type} ")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    bsz = "auto"
    
    model.to(DEV)
    
    if args.eval_ppl:
        datasets = ['wikitext2']

        for dataset in datasets:
            dataloader, testloader, tokenizer = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=2048
            )
            print(f"Evaluating {dataset} ...")
            ppl = eval_ppl(model, testloader)

            print(f"Result,{dataset},{ppl:.3f}")

    
            
    if args.tasks is not None:
        import random
        import lm_eval
        import numpy as np
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
        from transformers import AutoTokenizer

        if "llama" in args.model.lower():
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)

        task_manager = TaskManager()
        task_patterns = []
        for item in args.tasks:
            task_patterns.extend([x.strip() for x in str(item).split(",") if x.strip()])

        # Task aliases: ceval → ceval-valid (lm_eval registered name)
        TASK_ALIASES = {"ceval": "ceval-valid"}
        task_patterns = [TASK_ALIASES.get(t, t) for t in task_patterns]

        # Route longbench to its own script
        if "longbench" in task_patterns:
            task_patterns = [t for t in task_patterns if t != "longbench"]
            print("\nNote: 'longbench' is not in lm_eval. Evaluate separately with:")
            print("  python model/eval_longbench.py <same model/quant args> --tasks <task_names>")

        import json as _json
        fewshot_overrides = _json.loads(args.fewshot_overrides) if args.fewshot_overrides else {}
        batch_size_overrides = _json.loads(args.batch_size_overrides) if args.batch_size_overrides else {}

        task_names = sorted(set(lm_eval_utils.pattern_match(task_patterns, task_manager.all_tasks)))
        results_by_task = {}
        for task_name in task_names:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            model.eval()
            # Use per-task batch size; prefix-match so {"mmlu": 2} covers mmlu_abstract_algebra etc.
            task_bsz = next(
                (v for k, v in batch_size_overrides.items()
                 if task_name == k or task_name.startswith(k + '_')),
                args.lm_eval_batch_size,
            )
            hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=task_bsz)
            task_fewshot = fewshot_overrides.get(task_name, args.num_fewshot)
            result = lm_eval.simple_evaluate(
                hflm,
                tasks=[task_name],
                num_fewshot=task_fewshot,
                batch_size=task_bsz,
            )
            results_by_task[task_name] = result.get("results", {}).get(task_name, {})
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for task, metrics in results_by_task.items():
            print(f"\n{task}:")
            for k, v in metrics.items():
                if "stderr" not in k:
                    print(f"  {k}: {v}")

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        summary_metrics = {}
        for task, metrics in results_by_task.items():
            for k, v in metrics.items():
                if "stderr" in k:
                    continue
                if k.endswith("/acc") or "acc" in k.lower():
                    key = f"{task} {k}"
                    summary_metrics[key] = v
                    print(f"{key}: {v}")

        if hasattr(args, "wandb") and args.wandb and summary_metrics:
            import wandb
            wandb.log(summary_metrics)

        if args.output_file:
            import json
            from pathlib import Path
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(results_by_task, f, indent=2)
            print(f"\nResults saved to {args.output_file}")
  
