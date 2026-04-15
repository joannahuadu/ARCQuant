from datasets import load_dataset
import torch.nn as nn
import gc
from utilize import * 
import torch
from collections import defaultdict
import functools
from typing import List
import time
import pandas as pd
import numpy as np
import tqdm
import argparse
import math
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="path of the hf model")
parser.add_argument(
    "--dataset", type=str, default="wikitext2", choices=[
        "wikitext2",
        "c4",
        "humaneval",
        "pile",
        "arc_challenge",
        "arc_mix",
        "wikitext2_c4_mix_1to1",
        "wikitext2_c4_mix_3to1",
    ], 
    help="The calibration dataset to use."
)
parser.add_argument("--act_sort_metric", type=str, help="the metric used to sort the activations.")
parser.add_argument("--samples", type=int, default=128)
parser.add_argument("--seqlen", type=int, default=2048)


args = parser.parse_args()


def get_arc_challenge(nsamples, seed, seqlen, tokenizer):
    import random

    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="train")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    trainloader = []
    inps = []

    for idx in indices:
        sample = dataset[idx]
        question = str(sample.get("question", "")).strip()
        choices = sample.get("choices", {}) or {}
        labels = choices.get("label", []) or []
        texts = choices.get("text", []) or []

        lines = [f"Question: {question}", "Choices:"]
        for label, text in zip(labels, texts):
            lines.append(f"{label}. {str(text).strip()}")
        lines.append("Answer:")
        prompt = "\n".join(lines)

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=seqlen,
        ).input_ids
        if encoded.shape[1] < 8:
            continue

        inp = encoded[:, -seqlen:]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)

        if len(trainloader) >= nsamples:
            break

    return trainloader, inps


def get_arc_mix(nsamples, seed, seqlen, tokenizer):
    import random
    from pathlib import Path
    from datasets import Dataset, load_dataset

    def _load_cached_arrow(path_str):
        path = Path(path_str).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"missing cached dataset file: {path}")
        return Dataset.from_file(str(path))

    def _load_split(dataset_name, config_name, split, cache_path):
        try:
            return load_dataset(dataset_name, config_name, split=split)
        except Exception:
            return _load_cached_arrow(cache_path)

    random.seed(seed)
    texts = []

    arc_c = _load_split(
        "allenai/ai2_arc",
        "ARC-Challenge",
        "train",
        "~/.cache/huggingface/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-train.arrow",
    )
    for ex in arc_c:
        labels = ex["choices"]["label"]
        texts_choices = ex["choices"]["text"]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts_choices))
        texts.append(f"Question: {ex['question']}\n{opts}\nAnswer: {ex['answerKey']}")

    arc_e = _load_split(
        "allenai/ai2_arc",
        "ARC-Easy",
        "train",
        "~/.cache/huggingface/datasets/allenai___ai2_arc/ARC-Easy/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-train.arrow",
    )
    for ex in arc_e:
        labels = ex["choices"]["label"]
        texts_choices = ex["choices"]["text"]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts_choices))
        texts.append(f"Question: {ex['question']}\n{opts}\nAnswer: {ex['answerKey']}")

    rte = _load_split(
        "super_glue",
        "rte",
        "train",
        "~/.cache/huggingface/datasets/super_glue/rte/0.0.0/3de24cf8022e94f4ee4b9d55a6f539891524d646/super_glue-train.arrow",
    )
    label_map = {0: "yes", 1: "no"}
    for ex in rte:
        texts.append(
            f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
            f"Entailment: {label_map.get(ex['label'], 'yes')}"
        )

    random.shuffle(texts)
    trainenc = tokenizer("\n\n".join(texts), return_tensors="pt")

    trainloader = []
    inps = []
    for _ in range(nsamples):
        max_start = trainenc.input_ids.shape[1] - seqlen
        i = random.randint(0, max_start) if max_start > 0 else 0
        inp = trainenc.input_ids[:, i:i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)

    return trainloader, inps


def get_wikitext2_c4_mix(nsamples, seed, seqlen, tokenizer, *, wt_ratio, c4_ratio):
    from pathlib import Path
    from datasets import Dataset, concatenate_datasets
    import random

    def _load_c4_from_cache(split: str):
        cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "allenai___c4"
        if split == "train":
            base = cache_root / "default-b04fc8a0b8562884" / "0.0.0" / "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
            arrow_files = sorted(base.glob("c4-train-*.arrow"))
        else:
            raise ValueError(f"unsupported split: {split}")

        if not arrow_files:
            raise FileNotFoundError(f"no cached C4 arrow files found for split={split} under {base}")

        datasets = [Dataset.from_file(str(p)) for p in arrow_files]
        return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    wt_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    try:
        c4_train = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    except Exception:
        c4_train = _load_c4_from_cache("train")

    wt_texts = [t for t in wt_train["text"] if t and t.strip()]
    c4_texts = [t for t in c4_train["text"] if t and t.strip()]
    if not wt_texts or not c4_texts:
        raise ValueError("wikitext2 or c4 train split is empty after filtering blank texts")

    random.seed(seed)
    mixed_texts = []
    for _ in range(nsamples):
        for _ in range(wt_ratio):
            mixed_texts.append(random.choice(wt_texts))
        for _ in range(c4_ratio):
            mixed_texts.append(random.choice(c4_texts))
    random.shuffle(mixed_texts)

    trainenc = tokenizer("\n\n".join(mixed_texts), return_tensors="pt")
    if trainenc.input_ids.shape[1] < 8:
        raise ValueError("tokenized mixed corpus is unexpectedly too short")

    trainloader = []
    inps = []
    for _ in range(nsamples):
        max_start = trainenc.input_ids.shape[1] - seqlen
        i = random.randint(0, max_start) if max_start > 0 else 0
        inp = trainenc.input_ids[:, i:i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)

    return trainloader, inps


DATASET_LOADERS = {
    "wikitext2": get_wikitext2,
    "c4": get_c4,
    "pile": get_pile,
    "humaneval": get_humaneval,
    "arc_challenge": get_arc_challenge,
    "arc_mix": get_arc_mix,
    "wikitext2_c4_mix_1to1": lambda nsamples, seed, seqlen, tokenizer: get_wikitext2_c4_mix(
        nsamples, seed, seqlen, tokenizer, wt_ratio=1, c4_ratio=1
    ),
    "wikitext2_c4_mix_3to1": lambda nsamples, seed, seqlen, tokenizer: get_wikitext2_c4_mix(
        nsamples, seed, seqlen, tokenizer, wt_ratio=3, c4_ratio=1
    ),
}
        
def main():
    model, enc = load_model(args.model)
    folder_path = "./saved"
    path = args.model.rstrip('/')
    model_name = path.split('/')[-1]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    start_time = time.time()
    
    print(f"Using {args.dataset} dataset for calibration.")
    get_dataset = DATASET_LOADERS[args.dataset]

    dataset_name = args.dataset.lower()
    act_scales_filename = f'./saved/{model_name.lower()}_act_scales_{dataset_name}_{args.act_sort_metric}.pt'
    act_scores_filename = f'./saved/{model_name.lower()}_act_scores_{dataset_name}_{args.act_sort_metric}.pt'

    print("Getting activation stats...")
    if not os.path.exists(act_scales_filename):
        print("Generating activation stats...")
        dataloader, _ = get_dataset(
            nsamples=args.samples, seed=0, seqlen=args.seqlen, tokenizer=enc
        )

        act_scales = get_act_stats(
            model, dataloader, "cuda:0", metric=args.act_sort_metric, seqlen=args.seqlen
        )
        torch.save(act_scales, act_scales_filename)
        del dataloader
    else:
        print("Loading pre-saved activation stats...")
        act_scales = torch.load(act_scales_filename)
        

    print("Getting reording index...")
    reorder_index = get_reorder_index(model, act_scales, metric=args.act_sort_metric)
    
    print("Getting proportions...")

    _, inps = get_dataset(
                nsamples=32, seed=0, tokenizer=enc, seqlen=args.seqlen
            )
    select_num, average_bits = search_select_proportions(model, inps, "cuda", args.seqlen, reorder_index)
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    reorder_filename = f'./saved/{model_name.lower()}_reorder_index_{dataset_name}_{args.act_sort_metric}.pt'
    select_num_filename = f'./saved/{model_name.lower()}_select_num_{dataset_name}_{args.act_sort_metric}.pt'
    avg_bits_filename = f'./saved/{model_name.lower()}_average_bits_{dataset_name}_{args.act_sort_metric}.pt'

    print(f"Saving reorder index to {reorder_filename}")
    torch.save(reorder_index, reorder_filename)
    print(f"Saving select num to {select_num_filename}")
    torch.save(select_num, select_num_filename)
    print(f"Saving average bits to {avg_bits_filename}")
    torch.save(average_bits, avg_bits_filename)
    
if __name__ == "__main__":
    main()