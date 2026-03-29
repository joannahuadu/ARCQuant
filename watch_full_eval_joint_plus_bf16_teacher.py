#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/data/shichao/ARCQuant")
MD = BASE / "skill" / "full_eval_joint_plus_bf16_teacher.md"
QUEUE_LOG = BASE / "results" / "logs" / "run_full_eval_joint_plus_bf16_teacher.log"

JP_LM = BASE / "results" / "joint_plus_lm_eval.json"
JP_LB = BASE / "results" / "joint_plus_longbench.json"
BF_LM = BASE / "results" / "joint_bf16_teacher_lm_eval.json"
BF_LB = BASE / "results" / "joint_bf16_teacher_longbench.json"

JP_LM_LOG = BASE / "results" / "logs" / "joint_plus_lm_eval.log"
JP_LB_LOG = BASE / "results" / "logs" / "joint_plus_longbench.log"
BF_LM_LOG = BASE / "results" / "logs" / "joint_bf16_teacher_lm_eval.log"
BF_LB_LOG = BASE / "results" / "logs" / "joint_bf16_teacher_longbench.log"

LM_TASKS = [
    ("arc_challenge", "acc_norm"),
    ("arc_easy", "acc"),
    ("boolq", "acc"),
    ("mmlu", "acc"),
    ("ceval-valid", "acc"),
    ("openbookqa", "acc_norm"),
    ("piqa", "acc"),
    ("rte", "acc"),
    ("winogrande", "acc"),
]

LONGBENCH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]


def run(cmd: str) -> str:
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=BASE,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def proc_running(output_name: str) -> bool:
    cmd = f"ps -ef | rg -q '{re.escape(output_name)}'"
    return subprocess.run(["bash", "-lc", cmd], cwd=BASE).returncode == 0


def job_status(path: Path, output_name: str) -> str:
    if exists_and_nonempty(path):
        return "✅ done"
    if proc_running(output_name):
        return "🔄 running"
    return "⏳ pending"


def load_json(path: Path):
    if not exists_and_nonempty(path):
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def extract_lm_metrics(path: Path) -> dict[str, str]:
    data = load_json(path)
    out: dict[str, str] = {}
    if not isinstance(data, dict):
        return out
    for task, metric in LM_TASKS:
        task_data = data.get(task)
        if not isinstance(task_data, dict):
            out[task] = "TBD"
            continue
        key = "acc_norm,none" if metric == "acc_norm" else "acc,none"
        val = task_data.get(key)
        out[task] = f"{float(val):.4f}" if isinstance(val, (int, float)) else "TBD"
    return out


def extract_longbench_metrics(path: Path) -> dict[str, str]:
    data = load_json(path)
    out: dict[str, str] = {}
    if not isinstance(data, dict):
        return out
    results = data.get("results", {})
    if not isinstance(results, dict):
        results = {}
    for task in LONGBENCH_TASKS:
        val = results.get(task)
        out[task] = f"{float(val):.4f}" if isinstance(val, (int, float)) else "TBD"
    avg = data.get("average")
    out["average"] = f"{float(avg):.4f}" if isinstance(avg, (int, float)) else "TBD"
    return out


def last_log_lines(path: Path, n: int = 8) -> str:
    if not path.exists():
        return "TBD"
    lines = [line.rstrip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return "TBD"
    return "\n".join(lines[-n:])


def active_processes_text() -> str:
    out = run(
        "ps -ef | rg "
        "'joint_plus_lm_eval.json|joint_plus_longbench.json|joint_bf16_teacher_lm_eval.json|joint_bf16_teacher_longbench.json' "
        "| rg -v 'rg '"
    )
    return out if out else "TBD"


def replace_block(text: str, start: str, end: str, body: str) -> str:
    pattern = rf"({re.escape(start)}\n)(.*?)(\n{re.escape(end)})"
    return re.sub(pattern, rf"\1{body}\3", text, flags=re.S)


def update_md() -> None:
    text = MD.read_text()

    statuses = {
        "joint_plus lm_eval": job_status(JP_LM, "joint_plus_lm_eval.json"),
        "joint_plus LongBench": job_status(JP_LB, "joint_plus_longbench.json"),
        "joint_bf16_teacher lm_eval": job_status(BF_LM, "joint_bf16_teacher_lm_eval.json"),
        "joint_bf16_teacher LongBench": job_status(BF_LB, "joint_bf16_teacher_longbench.json"),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status_body = "\n".join(
        [
            f"**记录时间**：{timestamp}",
            "",
            "| Job | 状态 |",
            "|-----|------|",
            f"| joint_plus lm_eval | {statuses['joint_plus lm_eval']} |",
            f"| joint_plus LongBench | {statuses['joint_plus LongBench']} |",
            f"| joint_bf16_teacher lm_eval | {statuses['joint_bf16_teacher lm_eval']} |",
            f"| joint_bf16_teacher LongBench | {statuses['joint_bf16_teacher LongBench']} |",
            "",
            "### 当前可见进程",
            "",
            "```text",
            active_processes_text(),
            "```",
            "",
            "### 调度日志摘录",
            "",
            "```text",
            last_log_lines(QUEUE_LOG, 12),
            "```",
        ]
    )
    text = replace_block(text, "<!-- AUTO_STATUS_START -->", "<!-- AUTO_STATUS_END -->", status_body)

    jp_lm = extract_lm_metrics(JP_LM)
    bf_lm = extract_lm_metrics(BF_LM)
    lm_lines = [
        "| Task | Metric | joint_plus | joint_bf16_teacher |",
        "|------|--------|------------|--------------------|",
    ]
    for task, metric in LM_TASKS:
        lm_lines.append(
            f"| {task} | {metric} | {jp_lm.get(task, 'TBD')} | {bf_lm.get(task, 'TBD')} |"
        )
    text = replace_block(
        text,
        "<!-- AUTO_LM_EVAL_START -->",
        "<!-- AUTO_LM_EVAL_END -->",
        "\n".join(lm_lines),
    )

    jp_lb = extract_longbench_metrics(JP_LB)
    bf_lb = extract_longbench_metrics(BF_LB)
    lb_lines = [
        "| Task | joint_plus | joint_bf16_teacher |",
        "|------|------------|--------------------|",
    ]
    for task in LONGBENCH_TASKS + ["average"]:
        lb_lines.append(f"| {task} | {jp_lb.get(task, 'TBD')} | {bf_lb.get(task, 'TBD')} |")
    text = replace_block(
        text,
        "<!-- AUTO_LONGBENCH_START -->",
        "<!-- AUTO_LONGBENCH_END -->",
        "\n".join(lb_lines),
    )

    MD.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=900)
    args = parser.parse_args()

    if args.once:
        update_md()
        return

    while True:
        update_md()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
