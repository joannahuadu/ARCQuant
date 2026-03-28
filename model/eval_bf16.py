"""bf16 reference evaluation (no quantization, no reordering)."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import lm_eval
from lm_eval import utils as lm_eval_utils
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--tasks", nargs="+", default=["arc_challenge"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--fewshot_overrides", type=str,
                        default='{"mmlu": 5, "ceval-valid": 5}')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    TASK_ALIASES = {"ceval": "ceval-valid"}
    task_patterns = []
    for item in args.tasks:
        task_patterns.extend([x.strip() for x in str(item).split(",") if x.strip()])
    task_patterns = [TASK_ALIASES.get(t, t) for t in task_patterns]
    fewshot_overrides = json.loads(args.fewshot_overrides)

    print("Loading bf16 model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    task_manager = TaskManager()
    task_names = sorted(set(lm_eval_utils.pattern_match(task_patterns, task_manager.all_tasks)))
    print(f"Tasks to evaluate: {task_names}")

    results_by_task = {}
    for task_name in task_names:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
        task_fewshot = fewshot_overrides.get(task_name, args.num_fewshot)
        print(f"\nEvaluating {task_name} ({task_fewshot}-shot)...")
        result = lm_eval.simple_evaluate(
            hflm,
            tasks=[task_name],
            num_fewshot=task_fewshot,
            batch_size=args.batch_size,
        )
        results_by_task[task_name] = result.get("results", {}).get(task_name, {})
        metrics = {k: v for k, v in results_by_task[task_name].items() if "stderr" not in k}
        print(f"  {task_name}: {metrics}")

    print("\n" + "=" * 60)
    for task, metrics in results_by_task.items():
        for k, v in metrics.items():
            if "stderr" not in k and "acc" in k.lower():
                print(f"  {task} {k}: {v:.4f}")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results_by_task, f, indent=2)
        print(f"\nSaved to {args.output_file}")
