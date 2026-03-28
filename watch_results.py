#!/usr/bin/env python3
"""
Poll result JSON files and update skill/eval_all.md when new results arrive.
Run: python watch_results.py
"""
import json, os, time, re
from pathlib import Path

BASE = Path("/data/shichao/ARCQuant")
RESULT_FILES = {
    "bf16":   BASE / "results/bf16_lm_eval.json",
    "nvfp4":  BASE / "results/nvfp4_lm_eval.json",
    "xmask":  BASE / "results/xmask_lm_eval.json",
    "joint":  BASE / "results/joint_lm_eval.json",
}
LB_FILES = {
    "bf16":  BASE / "results/bf16_longbench.json",
    "nvfp4": BASE / "results/nvfp4_longbench.json",
    "joint": BASE / "results/joint_longbench.json",
}
LOG_FILES = {
    "bf16": BASE / "results/logs/bf16.log",
    "nvfp4": BASE / "results/logs/nvfp4.log",
    "xmask_joint": BASE / "results/logs/xmask_joint.log",
}
MARKDOWN = BASE / "skill/eval_all.md"

TASKS_ORDER = ["arc_challenge","arc_easy","boolq","mmlu","ceval-valid","openbookqa","piqa","rte","winogrande"]
LB_TASKS_ORDER = ["narrativeqa","hotpotqa","gov_report","triviaqa","samsum","passage_retrieval_en"]

def load_results():
    data = {}
    for config, path in RESULT_FILES.items():
        if path.exists():
            with open(path) as f:
                data[config] = json.load(f)
    return data

def load_lb_results():
    data = {}
    for config, path in LB_FILES.items():
        if path.exists():
            with open(path) as f:
                data[config] = json.load(f)
    return data

def get_current_task(log_path, config):
    """Read log to find current task being evaluated."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return "starting..."
    try:
        text = log_path.read_text()
        # For bf16: grep "Evaluating X"
        matches = re.findall(r'Evaluating (\S+)', text)
        if matches:
            return matches[-1]
        # For main.py: look at lm_eval task markers
        matches = re.findall(r'Overwriting default num_fewshot of (\S+)', text)
        if matches:
            return matches[-1]
        return "loading..."
    except:
        return "?"

def fmt_val(v, is_pct=False):
    if v is None or v == "TBD":
        return "TBD"
    try:
        fv = float(v)
        if is_pct:
            return f"{fv*100:.2f}%"
        return f"{fv:.4f}"
    except:
        return str(v)

def get_metric(task_data, key):
    if task_data is None:
        return None
    return task_data.get(key)

def summarize_results(results):
    """Print a quick summary table."""
    configs = ["bf16", "nvfp4", "xmask", "joint"]
    print(f"\n{'Task':<20} {'bf16':>8} {'nvfp4':>8} {'xmask':>8} {'joint':>8}  (acc/acc_norm)")
    print("-" * 70)
    for task in TASKS_ORDER:
        row = f"{task:<20}"
        for cfg in configs:
            d = results.get(cfg, {}).get(task, {})
            acc = d.get("acc,none") or d.get("acc_norm,none")
            if acc is not None:
                row += f" {float(acc):>8.4f}"
            else:
                row += f" {'TBD':>8}"
        print(row)

def update_markdown(results, lb_results):
    """Update skill/eval_all.md with completed results."""
    content = MARKDOWN.read_text()
    configs = ["bf16", "nvfp4", "xmask", "joint"]

    # Update lm_eval results tables
    for task in TASKS_ORDER:
        for metric_key, col_name in [("acc,none", "acc"), ("acc_norm,none", "acc_norm")]:
            values = {}
            for cfg in configs:
                d = results.get(cfg, {}).get(task, {})
                val = d.get(metric_key)
                if val is not None:
                    values[cfg] = float(val)

            if not values:
                continue

            # Replace TBD in the markdown table for this task + metric
            # Pattern: | arc_challenge | ... | TBD | ...
            # This is a simplified update - just update TBD values
            for cfg, val in values.items():
                # Find the table row for this task and update the config column
                pass  # handled below via full rebuild

    # Rebuild the acc table section
    acc_rows = []
    accn_rows = []
    for task in TASKS_ORDER:
        acc_row = [task]
        accn_row = [task]
        has_acc = False
        has_accn = False
        for cfg in configs:
            d = results.get(cfg, {}).get(task, {})
            acc = d.get("acc,none")
            accn = d.get("acc_norm,none")
            acc_row.append(f"{float(acc):.4f}" if acc is not None else "TBD")
            accn_row.append(f"{float(accn):.4f}" if accn is not None else "TBD")
            if acc is not None: has_acc = True
            if accn is not None: has_accn = True
        acc_rows.append(acc_row)
        accn_rows.append(accn_row)

    # Compute averages
    avg_row_acc = ["**Average**"]
    avg_row_accn = ["**Average**"]
    for cfg in configs:
        acc_vals = [float(results[cfg][t]["acc,none"]) for t in TASKS_ORDER
                    if cfg in results and t in results[cfg] and "acc,none" in results[cfg][t]]
        accn_vals = [float(results[cfg][t]["acc_norm,none"]) for t in TASKS_ORDER
                     if cfg in results and t in results[cfg] and "acc_norm,none" in results[cfg][t]]
        avg_row_acc.append(f"{sum(acc_vals)/len(acc_vals):.4f}" if acc_vals else "TBD")
        avg_row_accn.append(f"{sum(accn_vals)/len(accn_vals):.4f}" if accn_vals else "TBD")

    # Build updated acc table
    header = "| Task | Few-shot | bf16 | nvfp4 | xmask | joint | Δ(joint vs bf16) | Δ(joint vs nvfp4) |"
    sep    = "|------|----------|------|-------|-------|-------|-------------------|-------------------|"
    fewshot_map = {"arc_challenge":"0","arc_easy":"0","boolq":"0","mmlu":"5",
                   "ceval-valid":"5","openbookqa":"0","piqa":"0","rte":"0","winogrande":"0"}

    new_acc_lines = [header, sep]
    for row in acc_rows + [avg_row_acc]:
        task = row[0]
        fs = fewshot_map.get(task, "0") if task != "**Average**" else "—"
        vals = row[1:]  # bf16, nvfp4, xmask, joint
        bf16_v = vals[0] if vals[0] != "TBD" else None
        joint_v = vals[3] if vals[3] != "TBD" else None
        nvfp4_v = vals[1] if vals[1] != "TBD" else None
        delta_bf16 = f"{(float(joint_v)-float(bf16_v))*100:+.2f}%" if bf16_v and joint_v else "—"
        delta_nvfp4 = f"{(float(joint_v)-float(nvfp4_v))*100:+.2f}%" if nvfp4_v and joint_v else "—"
        new_acc_lines.append(f"| {task} | {fs} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {delta_bf16} | {delta_nvfp4} |")

    new_accn_lines = ["| Task | bf16 | nvfp4 | xmask | joint | Δ(joint vs bf16) | Δ(joint vs nvfp4) |",
                      "|------|------|-------|-------|-------|-------------------|-------------------|"]
    for row in accn_rows + [avg_row_accn]:
        task = row[0]
        vals = row[1:]
        bf16_v = vals[0] if vals[0] != "TBD" else None
        joint_v = vals[3] if vals[3] != "TBD" else None
        nvfp4_v = vals[1] if vals[1] != "TBD" else None
        delta_bf16 = f"{(float(joint_v)-float(bf16_v))*100:+.2f}%" if bf16_v and joint_v else "—"
        delta_nvfp4 = f"{(float(joint_v)-float(nvfp4_v))*100:+.2f}%" if nvfp4_v and joint_v else "—"
        new_accn_lines.append(f"| {task} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {delta_bf16} | {delta_nvfp4} |")

    # Replace acc table in markdown (no DOTALL — prevent .* from spanning lines)
    acc_pattern = r'(\| Task \| Few-shot \|[^\n]*\n)(\|[-|]+\|\n)((?:\|[^\n]*\|\n)+)'
    new_acc_block = "\n".join(new_acc_lines) + "\n"
    content_new = re.sub(acc_pattern, new_acc_block, content, count=1)

    # Replace accn table
    accn_pattern = r'(\| Task \| bf16 \|[^\n]*\n)(\|[-|]+\|\n)((?:\|[^\n]*\|\n)+)'
    new_accn_block = "\n".join(new_accn_lines) + "\n"
    content_new = re.sub(accn_pattern, new_accn_block, content_new, count=1)

    # Update LongBench section
    if lb_results:
        lb_rows = []
        for task in LB_TASKS_ORDER:
            row = [task]
            for cfg in ["bf16", "nvfp4", "joint"]:
                val = lb_results.get(cfg, {}).get("results", {}).get(task)
                row.append(f"{float(val):.4f}" if val is not None else "TBD")
            lb_rows.append(row)

        lb_header = "| Task | Metric | bf16 | nvfp4 | joint | Δ(joint vs bf16) |"
        lb_sep    = "|------|--------|------|-------|-------|------------------|"
        lb_metric_map = {"narrativeqa":"F1","hotpotqa":"F1","gov_report":"ROUGE-L",
                         "triviaqa":"F1","samsum":"ROUGE-L","passage_retrieval_en":"Accuracy"}
        new_lb_lines = [lb_header, lb_sep]
        for row in lb_rows:
            task = row[0]
            metric = lb_metric_map.get(task, "?")
            bf16_v, nvfp4_v, joint_v = row[1], row[2], row[3]
            delta = f"{(float(joint_v)-float(bf16_v))*100:+.2f}%" if bf16_v != "TBD" and joint_v != "TBD" else "—"
            new_lb_lines.append(f"| {task} | {metric} | {bf16_v} | {nvfp4_v} | {joint_v} | {delta} |")

        lb_pattern = r'(\| Task \| Metric \|[^\n]*\n)(\|[-|]+\|\n)((?:\|[^\n]*\|\n)+)'
        new_lb_block = "\n".join(new_lb_lines) + "\n"
        content_new = re.sub(lb_pattern, new_lb_block, content_new, count=1)

    MARKDOWN.write_text(content_new)
    print(f"[{time.strftime('%H:%M:%S')}] Updated {MARKDOWN}")


def main():
    print(f"Watching results in {BASE}/results/")
    print("Will update skill/eval_all.md when JSON files appear.\n")
    seen = set()
    interval = 480  # 8 minutes

    while True:
        results = load_results()
        lb_results = load_lb_results()
        new = set(results.keys()) | set(lb_results.keys())

        if new != seen:
            seen = new
            print(f"\n[{time.strftime('%H:%M:%S')}] New results: {new}")
            summarize_results(results)
            try:
                update_markdown(results, lb_results)
            except Exception as e:
                print(f"  markdown update error: {e}")

        # Show live progress
        print(f"\n[{time.strftime('%H:%M:%S')}] Status:")
        for cfg, log_path in LOG_FILES.items():
            task = get_current_task(log_path, cfg)
            done = len(results.get(cfg.split("_")[0], {})) if cfg != "xmask_joint" else max(
                len(results.get("xmask", {})), len(results.get("joint", {})))
            print(f"  {cfg:<15}: {task}  (completed configs: {list(results.keys())})")

        time.sleep(interval)


if __name__ == "__main__":
    main()
