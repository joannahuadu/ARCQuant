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


def _build_lm_window_trainloader_from_texts(texts, nsamples, seed, seqlen, tokenizer):
    import random

    trainenc = tokenizer("\n\n".join(texts), return_tensors='pt')
    if trainenc.input_ids.shape[1] <= seqlen:
        raise ValueError(
            f"tokenized mixed corpus too short for seqlen={seqlen}: "
            f"{trainenc.input_ids.shape[1]}"
        )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, trainenc


def get_wikitext2_c4_mix(nsamples, seed, seqlen, model, tokenizer, *, wt_ratio, c4_ratio):
    """Build a simple text-level mixture of wikitext2 and C4.

    The ratio is enforced by repeating/concatenating texts before tokenization.
    Validation follows wikitext2 so perplexity remains comparable to the main line.
    """
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

    wt_train = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    wt_test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    try:
        c4_train = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
    except Exception:
        c4_train = _load_c4_from_cache("train")

    wt_texts = [t for t in wt_train['text'] if t and t.strip()]
    c4_texts = [t for t in c4_train['text'] if t and t.strip()]
    if not wt_texts or not c4_texts:
        raise ValueError("wikitext2 or c4 train split is empty after filtering blank texts")

    import random
    random.seed(seed)

    mixed_texts = []
    total_parts = wt_ratio + c4_ratio
    for _ in range(nsamples):
        for _ in range(wt_ratio):
            mixed_texts.append(random.choice(wt_texts))
        for _ in range(c4_ratio):
            mixed_texts.append(random.choice(c4_texts))
    random.shuffle(mixed_texts)

    trainloader, _ = _build_lm_window_trainloader_from_texts(
        mixed_texts, nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer
    )
    testenc = tokenizer("\n\n".join(wt_test['text']), return_tensors='pt')
    return trainloader, testenc, tokenizer

def _load_ceval_texts():
    """Load all CEVAL dev+val splits across all subjects as formatted Chinese MCQ text."""
    from datasets import Dataset

    cache_base = Path.home() / ".cache" / "huggingface" / "datasets" / "ceval___ceval-exam"
    hash_dir = "0.0.0/617524a00b307ff6f9933702f724131fe12ca7ce"

    texts = []
    for subj_dir in sorted(cache_base.iterdir()):
        if not subj_dir.is_dir():
            continue
        for split in ("dev", "val"):
            arrow = subj_dir / hash_dir / f"ceval-exam-{split}.arrow"
            if not arrow.exists():
                continue
            ds = Dataset.from_file(str(arrow))
            for ex in ds:
                q = ex.get("question", "").strip()
                if not q:
                    continue
                opts = "\n".join(
                    f"{k}. {ex[k]}" for k in ("A", "B", "C", "D") if ex.get(k, "").strip()
                )
                answer = ex.get("answer", "").strip()
                explanation = ex.get("explanation", "").strip()
                text = f"题目：{q}\n{opts}\n答案：{answer}"
                if explanation:
                    text += f"\n解析：{explanation}"
                texts.append(text)

    if not texts:
        raise FileNotFoundError(f"no CEVAL texts found under {cache_base}")
    return texts


def get_wikitext2_c4_zh_mix(nsamples, seed, seqlen, model, tokenizer, *, wt_ratio, c4_ratio, zh_ratio):
    """3-way text-level mixture of wikitext2, C4, and CEVAL Chinese (dev+val).

    zh_ratio controls the proportion of Chinese CEVAL examples mixed in.
    Validation follows wikitext2 so perplexity remains comparable to the main line.
    """
    from datasets import Dataset, concatenate_datasets, load_dataset
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

    wt_train = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    wt_test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    try:
        c4_train = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
    except Exception:
        c4_train = _load_c4_from_cache("train")

    wt_texts = [t for t in wt_train['text'] if t and t.strip()]
    c4_texts = [t for t in c4_train['text'] if t and t.strip()]
    zh_texts = _load_ceval_texts()

    random.seed(seed)
    mixed_texts = []
    for _ in range(nsamples):
        for _ in range(wt_ratio):
            mixed_texts.append(random.choice(wt_texts))
        for _ in range(c4_ratio):
            mixed_texts.append(random.choice(c4_texts))
        for _ in range(zh_ratio):
            mixed_texts.append(random.choice(zh_texts))
    random.shuffle(mixed_texts)

    trainloader, _ = _build_lm_window_trainloader_from_texts(
        mixed_texts, nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer
    )
    testenc = tokenizer("\n\n".join(wt_test['text']), return_tensors='pt')
    return trainloader, testenc, tokenizer


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
    
    if name == 'wikitext2_c4_mix_1to1':
        return get_wikitext2_c4_mix(
            nsamples, seed, seqlen, model, tokenizer, wt_ratio=1, c4_ratio=1
        )
    if name == 'wikitext2_c4_mix_3to1':
        return get_wikitext2_c4_mix(
            nsamples, seed, seqlen, model, tokenizer, wt_ratio=3, c4_ratio=1
        )
    if name == 'wikitext2_c4_zh_mix_1to1to1':
        return get_wikitext2_c4_zh_mix(
            nsamples, seed, seqlen, model, tokenizer, wt_ratio=1, c4_ratio=1, zh_ratio=1
        )
    if name == 'wikitext2_c4_zh_mix_2to2to1':
        return get_wikitext2_c4_zh_mix(
            nsamples, seed, seqlen, model, tokenizer, wt_ratio=2, c4_ratio=2, zh_ratio=1
        )
    if name == 'wikitext2_c4_zh_mix_3to3to2':
        return get_wikitext2_c4_zh_mix(
            nsamples, seed, seqlen, model, tokenizer, wt_ratio=3, c4_ratio=3, zh_ratio=2
        )
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
