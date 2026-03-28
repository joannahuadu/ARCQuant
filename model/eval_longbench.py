"""
LongBench evaluation for quantized models.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=2 conda run -n smoothquant python model/eval_longbench.py \\
        meta-llama/Llama-3.1-8B --quant_type NVFP4 --no_xw_reorder \\
        [--use_x_mask --x_mask_ckpt <path>] \\
        [--softmax_alpha_ckpt <path>] \\
        --tasks narrativeqa hotpotqa gov_report trec triviaqa samsum \\
        --output_file results/longbench_joint.json

Available tasks:
    Single-doc QA : narrativeqa qasper multifieldqa_en multifieldqa_zh
    Multi-doc QA  : hotpotqa 2wikimqa musique dureader
    Summarization : gov_report qmsum multi_news vcsum
    Few-shot      : trec triviaqa samsum lsht
    Synthetic     : passage_count passage_retrieval_en passage_retrieval_zh
    Code          : lcc repobench-p
"""

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from datautils import DEV
from model_utils import reorder_model_llama
from x_mask_utils import (
    iter_layer_x_mask_modules,
    load_x_mask_checkpoint,
    parse_layer_spec,
    set_layer_x_mask_alpha,
    set_layer_x_mask_eval_mode,
)
from softmax_alpha_utils import load_softmax_alpha_checkpoint


# ── Task configs ─────────────────────────────────────────────────────────────

DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely "
        "as you can, using a single phrase or sentence if possible. If the question cannot be "
        "answered based on the information in the article, write \"unanswerable\". If the question "
        "is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single "
        "phrase or sentence if possible. If the question cannot be answered based on the information "
        "in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "multifieldqa_zh": (
        "阅读以下文字并用中文简短回答：\n\n{context}\n\n"
        "现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n"
        "问题：{input}\n回答："
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Passages: {context}\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Passages: {context}\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Passages: {context}\n\nQuestion: {input}\nAnswer:"
    ),
    "dureader": (
        "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n"
        "请基于上述文章回答下面的问题。\n\n问题：{input}\n回答："
    ),
    "gov_report": (
        "You are given a report by a government agency. "
        "Write a one-page summary of the report.\n\nReport:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news passages.\n\n"
        "News: {context}\n\n"
        "Now, write a one-page summary of all the news passages above.\n\nSummary:"
    ),
    "vcsum": (
        "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n"
        "会议记录：\n{context}\n\n会议总结："
    ),
    "trec":               "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa":           "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum":             "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht":               "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The output format should only contain the number, such as 1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n{context}\n\n"
        "The following is an abstract.\n\n{input}\n\nThe answer is Paragraph "
    ),
    "passage_retrieval_zh": (
        "以下是若干段落文字，以及其中一个段落的摘要。请确定给出的摘要出自哪一段。\n\n{context}\n\n"
        "下面是一段摘要\n\n{input}\n\n请输出摘要所属段落的编号。答案是第 "
    ),
    "lcc":          "Please complete the code given below.\n{context}Next line of code:\n",
    "repobench-p":  "Please complete the code given below.\n{context}{input}Next line of code:\n",
}

DATASET2MAXGEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "multifieldqa_zh": 64,
    "hotpotqa": 32,     "2wikimqa": 32, "musique": 32,          "dureader": 128,
    "gov_report": 512,  "qmsum": 512,   "multi_news": 512,      "vcsum": 512,
    "trec": 64,         "triviaqa": 32, "samsum": 128,           "lsht": 64,
    "passage_count": 32, "passage_retrieval_en": 32, "passage_retrieval_zh": 32,
    "lcc": 64,          "repobench-p": 64,
}

DATASET2METRIC = {
    "narrativeqa": "f1",       "qasper": "f1",           "multifieldqa_en": "f1",
    "multifieldqa_zh": "f1",   "hotpotqa": "f1",          "2wikimqa": "f1",
    "musique": "f1",           "dureader": "rouge",       "gov_report": "rouge",
    "qmsum": "rouge",          "multi_news": "rouge",     "vcsum": "rouge",
    "trec": "accuracy",        "triviaqa": "f1",          "samsum": "rouge",
    "lsht": "accuracy",        "passage_count": "accuracy",
    "passage_retrieval_en": "accuracy", "passage_retrieval_zh": "accuracy",
    "lcc": "code_sim",         "repobench-p": "code_sim",
}

ALL_TASKS = list(DATASET2PROMPT.keys())


# ── Metrics ──────────────────────────────────────────────────────────────────

def _normalize(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def compute_f1(prediction, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    best = 0.0
    pred_tokens = _normalize(prediction).split()
    for gt in ground_truths:
        gt_tokens = _normalize(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if not pred_tokens or not gt_tokens:
            score = float(pred_tokens == gt_tokens)
        elif num_same == 0:
            score = 0.0
        else:
            p = num_same / len(pred_tokens)
            r = num_same / len(gt_tokens)
            score = 2 * p * r / (p + r)
        best = max(best, score)
    return best


def _lcs_length(a, b):
    """Length of longest common subsequence (space-optimized DP)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def compute_rouge_l(prediction, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    try:
        from rouge_score import rouge_scorer as rs_mod
        scorer = rs_mod.RougeScorer(["rougeL"], use_stemmer=False)
        return max(scorer.score(gt, prediction)["rougeL"].fmeasure for gt in ground_truths)
    except ImportError:
        best = 0.0
        pred_tokens = _normalize(prediction).split()
        for gt in ground_truths:
            gt_tokens = _normalize(gt).split()
            if not pred_tokens or not gt_tokens:
                continue
            l = _lcs_length(pred_tokens, gt_tokens)
            p = l / len(pred_tokens)
            r = l / len(gt_tokens)
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            best = max(best, f)
        return best


def compute_accuracy(prediction, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    pred_norm = _normalize(prediction)
    return float(any(_normalize(gt) in pred_norm for gt in ground_truths))


def compute_code_sim(prediction, ground_truths):
    """Edit-distance similarity on the first line of code."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    def edit_sim(a, b):
        if not a and not b:
            return 1.0
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            prev, dp[0] = dp[0], i
            for j, cb in enumerate(b, 1):
                prev, dp[j] = dp[j], prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
        return 1.0 - dp[len(b)] / max(len(a), len(b))

    pred_line = prediction.strip().split("\n")[0]
    return max(edit_sim(pred_line, gt.strip().split("\n")[0]) for gt in ground_truths)


def score_prediction(prediction, ground_truths, metric):
    if metric == "f1":
        return compute_f1(prediction, ground_truths)
    elif metric == "rouge":
        return compute_rouge_l(prediction, ground_truths)
    elif metric == "accuracy":
        return compute_accuracy(prediction, ground_truths)
    elif metric == "code_sim":
        return compute_code_sim(prediction, ground_truths)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ── Generation ───────────────────────────────────────────────────────────────

def build_prompt(example, dataset_name):
    template = DATASET2PROMPT[dataset_name]
    context = example.get("context", "")
    input_ = example.get("input", "")
    return template.format(context=context, input=input_)


def generate_answer(model, tokenizer, prompt, max_new_tokens, max_input_length, device):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc.input_ids
    # Truncate: keep first half and last half to preserve context boundaries
    if input_ids.shape[1] > max_input_length:
        half = max_input_length // 2
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Per-task evaluation ──────────────────────────────────────────────────────

def evaluate_task(model, tokenizer, dataset_name, max_input_length, device, verbose=False):
    print(f"  Loading dataset '{dataset_name}'...")
    data = load_dataset("THUDM/LongBench", dataset_name, split="test", trust_remote_code=True)
    metric = DATASET2METRIC[dataset_name]
    max_gen = DATASET2MAXGEN[dataset_name]
    scores = []
    t0 = time.time()
    for i, example in enumerate(data):
        prompt = build_prompt(example, dataset_name)
        prediction = generate_answer(model, tokenizer, prompt, max_gen, max_input_length, device)
        answers = example.get("answers", example.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]
        sc = score_prediction(prediction, answers, metric)
        scores.append(sc)
        if verbose and (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    [{dataset_name}] {i+1}/{len(data)}  avg={np.mean(scores):.4f}  ({elapsed:.0f}s)")
    return float(np.mean(scores))


# ── Model loading (mirrors main.py) ──────────────────────────────────────────

def get_llama(model_path):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    return LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongBench evaluation for quantized models.")
    parser.add_argument("model", type=str, help="Model path (HuggingFace or local).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--act_sort_metric", type=str, default="max",
                        choices=["mean", "frobenius", "hessian", "max"])
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4", "pile", "humaneval"],
                        help="Calibration dataset used when generating reorder indices.")
    parser.add_argument("--quant_type", type=str, default="NVFP4",
                        choices=["NVFP4", "MXFP4", "INT4", "HiF4"])
    parser.add_argument("--kv_cache", action="store_true")
    parser.add_argument("--no_xw_reorder", action="store_true")
    parser.add_argument("--use_x_mask", action="store_true")
    parser.add_argument("--x_mask_ckpt", type=str, default=None)
    parser.add_argument("--x_mask_tau", type=float, default=1.0)
    parser.add_argument("--x_mask_alpha", type=float, default=1.0)
    parser.add_argument("--x_mask_skip_layers", type=str, default="")
    parser.add_argument("--x_mask_r_thr", type=float, default=-1.0)
    parser.add_argument("--rec", action="store_true")
    parser.add_argument("--softmax_alpha_ckpt", type=str, default=None)
    parser.add_argument("--softmax_alpha_skip_layers", type=str, default="")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        help=f"LongBench tasks. Choices: {ALL_TASKS}")
    parser.add_argument("--max_input_length", type=int, default=3900,
                        help="Max input tokens; longer inputs are truncated from the middle.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save results as JSON.")
    parser.add_argument("--no_quant", action="store_true",
                        help="Skip quantization; load bf16 model directly (for reference evaluation).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-example progress every 20 samples.")
    args = parser.parse_args()

    # ── Load and (optionally) quantize model ──
    if args.no_quant:
        print("Loading bf16 model (no quantization)...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
    else:
        model_name = args.model.rstrip("/").split("/")[-1]
        dataset_name = args.dataset.lower()
        index_filename = f"./saved/{model_name.lower()}_reorder_index_{dataset_name}_{args.act_sort_metric}.pt"
        select_num_filename = f"./saved/{model_name.lower()}_select_num_{dataset_name}_{args.act_sort_metric}.pt"
        act_scales_filename = f"./saved/{model_name.lower()}_act_scales_{dataset_name}_{args.act_sort_metric}.pt"
        assert os.path.isfile(index_filename), f"Reorder index not found: {index_filename}"

        print("Loading model...")
        model = get_llama(args.model)
        model.eval()

        print("Loading reorder index...")
        reorder_index = torch.load(index_filename, weights_only=False)
        select_nums   = torch.load(select_num_filename, weights_only=False)
        act_scales    = torch.load(act_scales_filename, weights_only=False)

        print("Reordering / quantizing model...")
        model = reorder_model_llama(
            model,
            device=DEV,
            kv_cache=args.kv_cache,
            reorder_index=reorder_index,
            select_nums=select_nums,
            quant_type=args.quant_type,
            reorder_xw=not args.no_xw_reorder,
            use_x_mask=args.use_x_mask,
            x_mask_tau=args.x_mask_tau,
            x_mask_alpha=args.x_mask_alpha,
            x_mask_skip_layers=args.x_mask_skip_layers,
            x_mask_r_thr=None if args.x_mask_r_thr < 0 else args.x_mask_r_thr,
            rec=args.rec,
        )
        model.eval()

    if not args.no_quant:
        if args.use_x_mask and args.x_mask_ckpt:
            meta = load_x_mask_checkpoint(model, args.x_mask_ckpt)
            if meta:
                print(f"Loaded x-mask ckpt: {meta}")

        if args.softmax_alpha_ckpt:
            meta = load_softmax_alpha_checkpoint(
                model, args.softmax_alpha_ckpt, skip_layers=args.softmax_alpha_skip_layers
            )
            if meta:
                print(f"Loaded softmax alpha ckpt: {meta}")

        if args.use_x_mask:
            skip_layers = parse_layer_spec(args.x_mask_skip_layers)
            x_mask_r_thr = None if args.x_mask_r_thr < 0 else args.x_mask_r_thr
            for layer_idx, layer in enumerate(model.model.layers):
                if layer_idx in skip_layers:
                    continue
                set_layer_x_mask_alpha(layer, args.x_mask_alpha)
                if x_mask_r_thr is not None:
                    for xm in iter_layer_x_mask_modules(layer):
                        xm.x_mask_r_thr = x_mask_r_thr
                    set_layer_x_mask_eval_mode(layer, True)

        model.to(DEV)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Filter valid tasks ──
    tasks = [t for t in args.tasks if t in DATASET2PROMPT]
    unknown = [t for t in args.tasks if t not in DATASET2PROMPT]
    if unknown:
        print(f"Warning: unknown LongBench tasks (skipped): {unknown}")
        print(f"Available: {ALL_TASKS}")

    # ── Run evaluation ──
    results = {}
    for task in tasks:
        print(f"\n[{task}] evaluating...")
        score = evaluate_task(model, tokenizer, task, args.max_input_length, DEV, args.verbose)
        results[task] = score
        print(f"  {task}: {score:.4f}")

    print("\n" + "=" * 60)
    print("LongBench Results")
    print("=" * 60)
    for task, score in results.items():
        print(f"  {task:<30s}: {score:.4f}")
    if results:
        avg = float(np.mean(list(results.values())))
        print(f"  {'Average':<30s}: {avg:.4f}")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(
                {"results": results, "average": float(np.mean(list(results.values())))},
                f, indent=2
            )
        print(f"\nResults saved to {args.output_file}")
