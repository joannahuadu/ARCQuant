import numpy as np
import torch
from pathlib import Path

DEV = torch.device('cuda:0')

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc, tokenizer

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc, tokenizer

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    from datasets import Dataset, concatenate_datasets, load_dataset

    def _load_c4_from_cache(split: str):
        cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "allenai___c4"
        if split == "train":
            base = cache_root / "default-b04fc8a0b8562884" / "0.0.0" / "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
            arrow_files = sorted(base.glob("c4-train-*.arrow"))
        elif split == "validation":
            base = cache_root / "default-c7bc8b0aefc5e48f" / "0.0.0" / "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
            arrow_files = sorted(base.glob("c4-validation*.arrow"))
        else:
            raise ValueError(f"unsupported split: {split}")

        if not arrow_files:
            raise FileNotFoundError(f"no cached C4 arrow files found for split={split} under {base}")

        datasets = [Dataset.from_file(str(p)) for p in arrow_files]
        return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    try:
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
    except Exception:
        traindata = _load_c4_from_cache("train")

    try:
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )
    except Exception:
        valdata = _load_c4_from_cache("validation")
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc, tokenizer

def get_ptb_new(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc, tokenizer

def get_c4_new(nsamples, seed, seqlen, model, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc, tokenizer


def get_arc_mix(nsamples, seed, seqlen, model, tokenizer):
    """Calibration data from arc_challenge + arc_easy + rte train splits.

    Mixes MCQ (arc) and NLI (rte) format text to improve generalization on
    downstream benchmarks with these task formats.
    """
    from datasets import Dataset, load_dataset
    import random

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

    # arc_challenge train
    arc_c = _load_split(
        'allenai/ai2_arc',
        'ARC-Challenge',
        'train',
        '~/.cache/huggingface/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-train.arrow',
    )
    for ex in arc_c:
        ch_labels = ex['choices']['label']
        ch_texts = ex['choices']['text']
        opts = "\n".join(f"{l}. {t}" for l, t in zip(ch_labels, ch_texts))
        texts.append(f"Question: {ex['question']}\n{opts}\nAnswer: {ex['answerKey']}")

    # arc_easy train
    arc_e = _load_split(
        'allenai/ai2_arc',
        'ARC-Easy',
        'train',
        '~/.cache/huggingface/datasets/allenai___ai2_arc/ARC-Easy/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-train.arrow',
    )
    for ex in arc_e:
        ch_labels = ex['choices']['label']
        ch_texts = ex['choices']['text']
        opts = "\n".join(f"{l}. {t}" for l, t in zip(ch_labels, ch_texts))
        texts.append(f"Question: {ex['question']}\n{opts}\nAnswer: {ex['answerKey']}")

    # rte train (SuperGLUE)
    rte = _load_split(
        'super_glue',
        'rte',
        'train',
        '~/.cache/huggingface/datasets/super_glue/rte/0.0.0/3de24cf8022e94f4ee4b9d55a6f539891524d646/super_glue-train.arrow',
    )
    label_map = {0: "yes", 1: "no"}
    for ex in rte:
        texts.append(
            f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
            f"Entailment: {label_map.get(ex['label'], 'yes')}"
        )

    random.shuffle(texts)
    all_text = "\n\n".join(texts)
    trainenc = tokenizer(all_text, return_tensors='pt')

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        inp = trainenc.input_ids[:, i:i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, trainenc, tokenizer


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    # assert "llama" in model.lower(), "Only llama models are supported."

    if "llama" in model.lower():
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
                print(f"bos/eos tokens updated: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
            except AttributeError:
                pass
                print(f"bos/eos tokens unchanged: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
    else:
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, legacy=False)
    
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model, tokenizer)
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model, tokenizer)
        return get_c4(nsamples, seed, seqlen, model, tokenizer)
    if 'arc_mix' in name:
        return get_arc_mix(nsamples, seed, seqlen, model, tokenizer)
