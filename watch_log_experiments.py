#!/usr/bin/env python3
"""
Monitor log.py softmax-stats experiments.
Usage: python watch_log_experiments.py [--once]
"""
import re, time, sys, os
from pathlib import Path

BASE = Path("/data/shichao/ARCQuant")

EXPERIMENTS = {
    "nvfp4":  {"log": BASE/"results/logs/log_nvfp4.log",      "pids": [785044]},
    "joint":  {"log": BASE/"results/logs/log_joint.log",       "pids": [785045]},
    "xmask":  {"log": BASE/"results/logs/log_xmask_bf16.log",  "pids": [788320]},
    "bf16":   {"log": BASE/"results/logs/log_xmask_bf16.log",  "pids": [788320]},
}

def is_alive(pid):
    return Path(f"/proc/{pid}").exists()

def read_tail(path, nbytes=4000):
    try:
        data = Path(path).read_bytes()
        return data[-nbytes:].decode("utf-8", errors="replace")
    except:
        return ""

def get_progress(text):
    """Extract latest loglikelihood progress bar."""
    matches = re.findall(r'(\d+)/(\d+)\s+\[[\d:]+<([\d:]+),\s+([\d.]+)it/s\]', text)
    if matches:
        done, total, eta, rate = matches[-1]
        return int(done), int(total), eta, float(rate)
    return None

def get_arc_result(text):
    matches = re.findall(r'arc_challenge.*?acc[,_]none["\s:]+([0-9.]+)', text)
    if matches:
        return float(matches[-1])
    return None

def get_xq_stats(text):
    """Extract last few x_q zero_ratio lines."""
    lines = re.findall(r'\[([^\]]+)\] zero_ratio=([\d.]+), two_zeros_ratio=([\d.]+)', text)
    return lines[-3:] if lines else []

def get_softmax_saved(exp_dir_pattern):
    """Check if softmax_stats.json was saved."""
    for d in sorted(BASE.glob(f"outputs/{exp_dir_pattern}*"))[-1:]:
        f = d / "softmax_stats.json"
        if f.exists():
            return str(f)
    return None

def get_exp_dir(text):
    m = re.search(r'exp_dir[= ](outputs/[^\s]+)', text)
    return m.group(1) if m else None

def current_phase(text):
    """Determine what phase the experiment is in."""
    if "softmax_stats] saved:" in text:
        return "DONE"
    if "arc_challenge" in text and re.search(r'acc[,_]none', text):
        return "arc_challenge done"
    if "Running loglikelihood" in text:
        prog = get_progress(text)
        if prog:
            done, total, eta, rate = prog
            pct = done * 100 // total
            return f"arc_challenge {pct}% ({done}/{total}, ETA {eta}, {rate:.1f}it/s)"
    if "Reordering model" in text or "reorder" in text.lower():
        return "loading/reordering"
    if "skip_quantize" in text:
        return "loading bf16"
    return "starting..."

def report():
    print(f"\n{'='*70}")
    print(f"  log.py experiment monitor  [{time.strftime('%H:%M:%S')} UTC]")
    print(f"{'='*70}")

    for name, cfg in [("nvfp4", EXPERIMENTS["nvfp4"]),
                       ("joint", EXPERIMENTS["joint"]),
                       ("xmask+bf16", EXPERIMENTS["xmask"])]:
        pid = cfg["pids"][0]
        alive = is_alive(pid)
        log = cfg["log"]
        text = read_tail(log)

        phase = current_phase(text)
        status = "ALIVE" if alive else "DEAD"

        print(f"\n  [{name}] PID={pid} [{status}]")
        print(f"    Phase : {phase}")

        xq = get_xq_stats(text)
        if xq:
            print(f"    x_q   : (last 3 modules)")
            for mod, zr, t24 in xq:
                short = mod.split(".")[-3] + "." + mod.split(".")[-1]
                print(f"            {short:<30}  zero={float(zr):.4f}  2:4={float(t24):.4f}")

        arc = get_arc_result(text)
        if arc:
            print(f"    arc_challenge acc: {arc:.4f}")

    # Check for completed output dirs
    print(f"\n  Output dirs:")
    for d in sorted(BASE.glob("outputs/log_*_arc_*")):
        stat_f = d / "softmax_stats.json"
        run_log = d / "run.log"
        has_stats = "stats_SAVED" if stat_f.exists() else "no_stats_yet"
        n_pngs = len(list((d / "act_dist").glob("*.png"))) if (d / "act_dist").exists() else 0
        print(f"    {d.name:<45} [{has_stats}] [{n_pngs} pngs]")

    print()

if __name__ == "__main__":
    once = "--once" in sys.argv
    while True:
        report()
        if once:
            break
        time.sleep(120)
