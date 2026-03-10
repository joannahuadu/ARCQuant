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
    set_layer_x_mask_alpha,
    set_layer_x_mask_eval_mode,
)


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
    parser.add_argument(
        "--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "pile", "humaneval"], 
        help="The calibration dataset to use."
    )
    parser.add_argument(
        "--quant_type", type=str, default="NVFP4", choices=["NVFP4", "MXFP4", "INT4", "HiF4"], 
        help="data type for W and A quantization."
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
    model = reorder_model_func(
        model,
        device=DEV,
        kv_cache=args.kv_cache,
        reorder_index=reorder_index,
        select_nums=select_nums,
        quant_type=args.quant_type,
        use_x_mask=bool(args.use_x_mask),
        x_mask_tau=float(args.x_mask_tau),
        x_mask_alpha=float(args.x_mask_alpha),
        x_mask_r_thr=None if float(args.x_mask_r_thr) < 0 else float(args.x_mask_r_thr),
    )
    end_time=time.time()
    peak_memory = torch.cuda.max_memory_allocated()

    if args.use_x_mask and args.x_mask_ckpt:
        meta = load_x_mask_checkpoint(model, args.x_mask_ckpt)
        if meta:
            print(f"Loaded x-mask ckpt meta: {meta}")

    if args.use_x_mask:
        x_mask_r_thr = None if float(args.x_mask_r_thr) < 0 else float(args.x_mask_r_thr)
        for layer in model.model.layers:
            set_layer_x_mask_alpha(layer, float(args.x_mask_alpha))
            if x_mask_r_thr is not None:
                for xm in iter_layer_x_mask_modules(layer):
                    xm.x_mask_r_thr = x_mask_r_thr
            if args.x_mask_eval_hard:
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
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import initialize_tasks
        initialize_tasks()

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

        task_names = lm_eval_utils.pattern_match(args.tasks, lm_eval.tasks.ALL_TASKS)
        results = lm_eval.simple_evaluate(
            hflm,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.lm_eval_batch_size,
        )

        results_by_task = results.get("results", {})
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
  