#!/usr/bin/env python3
"""Monitor P3 experiments and auto-update skill/p3_experiment_log.md.

Usage:
  python watch_p3_results.py --once
  nohup python watch_p3_results.py > results/logs/watch_p3_results.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


BASE = Path("/data/shichao/ARCQuant")
P3_MD = BASE / "skill" / "p3_experiment_log.md"
RESULTS = BASE / "results"
OUTPUTS = BASE / "outputs" / "Llama-3.1-8B" / "NVFP4"

CLIP_JSON = RESULTS / "joint_bf16_teacher_clip085_115_eval.json"
DEEPCLIP_JSON = RESULTS / "joint_bf16_teacher_deepclip085_115_l24_31_eval.json"
MIXED_EVAL_JSON = RESULTS / "joint_mixed_teacher_v2_eval.json"
MIXED_EXP_DIR = OUTPUTS / "joint_mixed_teacher_v2_20260328_093633"
MIXED_CKPT = MIXED_EXP_DIR / "llama-3.1-8b_joint_wikitext2_max_NVFP4.pt"

TASKS = ["arc_challenge", "rte", "ceval-valid"]


def load_metrics(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    out = {}
    for task in TASKS:
        metrics = data[task]
        key = "acc_norm,none" if "acc_norm,none" in metrics else "acc,none"
        out[task] = float(metrics[key])
    return out


def fmt(v: float) -> str:
    return f"{v:.4f}"


def delta(v: float, base: float) -> str:
    d = v - base
    return f"{d:+.2%}"


def update_md():
    text = P3_MD.read_text()

    clip = load_metrics(CLIP_JSON)
    deep = load_metrics(DEEPCLIP_JSON)
    mixed = load_metrics(MIXED_EVAL_JSON)

    clip_status = "✅ done" if clip else "🔄 running"
    deep_status = "✅ done" if deep else "🔄 running"
    text = re.sub(
        r"\| global clip \| .*? \| `results/joint_bf16_teacher_clip085_115_eval\.json` \|",
        f"| global clip | {clip_status} | `results/joint_bf16_teacher_clip085_115_eval.json` |",
        text,
    )
    text = re.sub(
        r"\| deep-layer clip \| .*? \| `results/joint_bf16_teacher_deepclip085_115_l24_31_eval\.json` \|",
        f"| deep-layer clip | {deep_status} | `results/joint_bf16_teacher_deepclip085_115_l24_31_eval.json` |",
        text,
    )

    if clip and deep:
        repl = (
            "| arc_challenge | 0.4881 | " + fmt(clip["arc_challenge"]) + " | " + fmt(deep["arc_challenge"]) + " |\n"
            "| rte | 0.7040 | " + fmt(clip["rte"]) + " | " + fmt(deep["rte"]) + " |\n"
            "| ceval-valid | 0.4324 | " + fmt(clip["ceval-valid"]) + " | " + fmt(deep["ceval-valid"]) + " |"
        )
        text = re.sub(
            r"\| arc_challenge \| 0\.4881 \| .*?\n\| rte \| 0\.7040 \| .*?\n\| ceval-valid \| 0\.4324 \| .*?\|",
            repl,
            text,
        )

    mixed_status = "✅ done" if mixed else "🔄 running"
    text = re.sub(r"\| 状态 \| .* \|", f"| 状态 | {mixed_status} |", text, count=1)

    if mixed:
        repl = (
            "| arc_challenge | 0.5316 | 0.5145 | 0.5111 | " + fmt(mixed["arc_challenge"]) + " |\n"
            "| rte | 0.7004 | 0.6859 | 0.6787 | " + fmt(mixed["rte"]) + " |\n"
            "| ceval-valid | 0.5245 | 0.4844 | 0.4510 | " + fmt(mixed["ceval-valid"]) + " |"
        )
        text = re.sub(
            r"\| arc_challenge \| 0\.5316 \| 0\.5145 \| 0\.5111 \| .*?\n\| rte \| 0\.7004 \| 0\.6859 \| 0\.6787 \| .*?\n\| ceval-valid \| 0\.5245 \| 0\.4844 \| 0\.4510 \| .*?\|",
            repl,
            text,
        )

    P3_MD.write_text(text)


def has_running_eval() -> bool:
    cmd = "ps -ef | rg 'joint_mixed_teacher_v2_eval.json|python model/main.py .*joint_mixed_teacher_v2_20260328_093633' -q"
    return subprocess.run(["bash", "-lc", cmd]).returncode == 0


def maybe_launch_mixed_eval():
    if not MIXED_CKPT.exists() or MIXED_EVAL_JSON.exists() or has_running_eval():
        return False
    cmd = (
        "CUDA_VISIBLE_DEVICES=3 conda run -n smoothquant python model/main.py "
        "/data/shichao/models/Llama-3.1-8B --quant_type NVFP4 --no_xw_reorder "
        f"--use_x_mask --x_mask_ckpt {MIXED_CKPT} --softmax_alpha_ckpt {MIXED_CKPT} "
        "--tasks arc_challenge rte ceval --lm_eval_batch_size 4 "
        f"--output_file {MIXED_EVAL_JSON}"
    )
    subprocess.Popen(["bash", "-lc", cmd], cwd=BASE)
    return True


def tick():
    launched = maybe_launch_mixed_eval()
    update_md()
    states = {
        "clip": CLIP_JSON.exists(),
        "deepclip": DEEPCLIP_JSON.exists(),
        "mixed_ckpt": MIXED_CKPT.exists(),
        "mixed_eval": MIXED_EVAL_JSON.exists(),
        "launched_mixed_eval": launched,
    }
    print(json.dumps(states, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=900, help="Poll interval in seconds; default 900 (15 min).")
    args = parser.parse_args()

    if args.once:
        tick()
        return

    while True:
        tick()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
