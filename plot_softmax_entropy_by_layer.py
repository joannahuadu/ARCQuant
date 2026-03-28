#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _resolve_stats_path(p: Path) -> Path:
    if p.is_dir():
        for name in ["softmax_stats.pt", "softmax_stats.json"]:
            cand = p / name
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No softmax_stats.(pt|json) found under directory: {p}")

    if p.suffix in [".pt", ".pth", ".bin", ".json"]:
        if p.exists():
            return p
        raise FileNotFoundError(p)

    for cand in [p.with_suffix(".pt"), p.with_suffix(".json")]:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Stats file not found: tried {p}.pt and {p}.json")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected payload type in {path}: {type(obj)} (expected dict)")
    # The per-layer summary is stored at the top-level in .pt state_dict.
    payload = dict(obj.get("summary") or {})
    payload["entropy_norm_by_layer_summary"] = obj.get("entropy_norm_by_layer_summary") or {}
    return payload


def _parse_layer_summary(payload: dict[str, Any]) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    by_layer = payload.get("entropy_norm_by_layer_summary") or {}
    if not isinstance(by_layer, dict) or not by_layer:
        raise ValueError("Missing entropy_norm_by_layer_summary (did you enable --softmax_stats_per_layer?)")

    layer_ids = sorted(int(k) for k in by_layer.keys())
    means: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    lt_001: list[float] = []
    for lid in layer_ids:
        item = by_layer[str(lid)]
        vmin = item.get("min", None)
        vmax = item.get("max", None)
        mins.append(float(vmin) if vmin is not None else float("nan"))
        maxs.append(float(vmax) if vmax is not None else float("nan"))
        means.append(float(item.get("mean", float("nan"))))
        lt_thresholds = item.get("lt_thresholds") or []
        lt_ratio = item.get("lt_ratio") or []
        # Prefer threshold==0.01 if present.
        try:
            j = list(lt_thresholds).index(0.01)
        except ValueError:
            j = 1 if len(lt_ratio) > 1 else 0
        lt_001.append(float(lt_ratio[j]) if j < len(lt_ratio) else float("nan"))
    return layer_ids, means, mins, maxs, lt_001


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-layer normalized entropy stats from SoftmaxStatsCollector.\n"
            "Requires --softmax_stats_per_layer when collecting."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Add one run (repeatable). Example: --run bf16 /path/to/softmax_stats",
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures/softmax_entropy_by_layer"))
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for label, p in args.run:
        path = _resolve_stats_path(Path(p))
        payload = _load_payload(path)
        layer_ids, means, mins, maxs, lt_001 = _parse_layer_summary(payload)
        runs.append((label, path, layer_ids, means, mins, maxs, lt_001))

    # Use layer ids from first run as x-axis reference.
    x = runs[0][2]
    for label, _path, layer_ids, _means, _mins, _maxs, _lt in runs[1:]:
        if layer_ids != x:
            raise ValueError(f"Layer id mismatch vs first run: {label} has {layer_ids[:5]}... (len={len(layer_ids)})")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True)
    for label, _path, _layer_ids, means, mins, maxs, lt_001 in runs:
        (line,) = ax1.plot(x, means, label=label, linewidth=1.8)
        # ax1.fill_between(
        #     x,
        #     mins,
        #     maxs,
        #     color=line.get_color(),
        #     alpha=0.18,
        #     linewidth=0.0,
        # )
        ax2.plot(x, lt_001, label=label, linewidth=1.8)

    ax1.set_title("Per-layer normalized entropy (mean with min-max band)")
    ax1.set_ylabel("H_norm")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.set_title("Per-layer low-entropy ratio (H_norm < 0.01)")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("ratio")
    ax2.grid(True, alpha=0.25)

    args.outdir.mkdir(parents=True, exist_ok=True)
    pref = f"{args.prefix}_" if args.prefix else ""
    outpath = args.outdir / f"{pref}entropy_norm_by_layer.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved: {outpath.resolve()}")


if __name__ == "__main__":
    main()
