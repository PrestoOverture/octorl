"""P1.24 — difficulty-bucket heatmap for the base model.

Two modes:
  --input baseline_eval.json   model pass@1 heatmap (primary)
  --coverage                   injector coverage heatmap (no model needed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

ACTIVE_BUCKETS = [
    "c1-L0-single-function", "c1-L1-single-function", "c1-L2-single-function",
    "c2-L0-single-function", "c2-L1-single-function", "c2-L2-single-function",
    "c3-L0-single-function", "c3-L1-single-function", "c3-L2-single-function",
    "c2-L0-single-file",     "c2-L1-single-file",     "c2-L2-single-file",
    "c3-L0-single-file",     "c3-L1-single-file",     "c3-L2-single-file",
]

TRAIN_REPOS = [
    "parcel_ledger", "record_index", "slot_planner",
    "config_parser", "metric_aggregator", "task_scheduler",
]
HELDOUT_REPOS = ["route_graph", "stock_reservations"]
ALL_REPOS = TRAIN_REPOS + HELDOUT_REPOS


def bucket_label(bid: str) -> str:
    parts = bid.split("-", 2)
    count = parts[0]
    hint = parts[1]
    span = parts[2].replace("single-", "s-").replace("cross-", "x-")
    return f"{count} {hint} {span}"


def heatmap_from_eval(data: dict, out: Path) -> None:
    pass_at_1 = data.get("pass_at_1", {})
    matrix = np.full((len(ACTIVE_BUCKETS), len(ALL_REPOS)), np.nan)
    annot = [[" "] * len(ALL_REPOS) for _ in range(len(ACTIVE_BUCKETS))]

    for i, bucket in enumerate(ACTIVE_BUCKETS):
        for j, repo in enumerate(ALL_REPOS):
            key = f"{repo}|{bucket}"
            if key in pass_at_1:
                val = pass_at_1[key]["mean_reward"]
                matrix[i, j] = val
                annot[i][j] = f"{val:.2f}"

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.isnan(matrix)
    sns.heatmap(
        matrix, mask=mask, ax=ax, vmin=0, vmax=1,
        cmap="YlOrRd_r", linewidths=0.5, linecolor="white",
        annot=annot, fmt="", annot_kws={"fontsize": 8},
        cbar_kws={"label": "pass@1", "shrink": 0.7},
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if mask[i, j]:
                ax.text(j + 0.5, i + 0.5, "N/A", ha="center", va="center",
                        fontsize=7, color="#999999")

    ax.set_xticklabels(
        [f"{r}\n(held-out)" if r in HELDOUT_REPOS else r for r in ALL_REPOS],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_yticklabels([bucket_label(b) for b in ACTIVE_BUCKETS],
                       rotation=0, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Difficulty Bucket")

    model_name = data.get("model", {}).get("model", "base model")
    rollouts = data.get("model", {}).get("rollouts_per_instance", "?")
    ax.set_title(f"Base Model pass@1 by Difficulty Bucket × Repository\n"
                 f"({model_name}, G={rollouts})", fontsize=12, pad=12)

    ax.axhline(y=9, color="black", linewidth=1.5)
    ax.text(-0.3, 4.5, "single-function", va="center", ha="right",
            fontsize=8, rotation=90, fontstyle="italic")
    ax.text(-0.3, 12, "single-file", va="center", ha="right",
            fontsize=8, rotation=90, fontstyle="italic")

    ax.axvline(x=len(TRAIN_REPOS), color="black", linewidth=1.5)

    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {out}")


def heatmap_from_coverage(out: Path) -> None:
    coverage_path = ROOT / "artifacts/p1/injector_coverage.json"
    if not coverage_path.exists():
        print(f"ERROR: {coverage_path} not found", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(coverage_path.read_text())
    cells = raw["cells"] if isinstance(raw, dict) else raw
    ok_counts: dict[tuple[str, str], int] = {}
    total_counts: dict[tuple[str, str], int] = {}
    for row in cells:
        bucket = row["bucket"]
        repo = row["repo"]
        if bucket not in ACTIVE_BUCKETS:
            continue
        key = (bucket, repo)
        total_counts[key] = total_counts.get(key, 0) + 1
        if row["status"] == "ok":
            ok_counts[key] = ok_counts.get(key, 0) + 1

    matrix = np.full((len(ACTIVE_BUCKETS), len(ALL_REPOS)), np.nan)
    annot = [[" "] * len(ALL_REPOS) for _ in range(len(ACTIVE_BUCKETS))]

    for i, bucket in enumerate(ACTIVE_BUCKETS):
        for j, repo in enumerate(ALL_REPOS):
            key = (bucket, repo)
            total = total_counts.get(key, 0)
            ok = ok_counts.get(key, 0)
            if total > 0:
                frac = ok / total
                matrix[i, j] = frac
                annot[i][j] = f"{ok}/{total}"

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.isnan(matrix)
    sns.heatmap(
        matrix, mask=mask, ax=ax, vmin=0, vmax=1,
        cmap="YlGn", linewidths=0.5, linecolor="white",
        annot=annot, fmt="", annot_kws={"fontsize": 7},
        cbar_kws={"label": "ok / total cells", "shrink": 0.7},
    )
    ax.set_xticklabels(
        [f"{r}\n(held-out)" if r in HELDOUT_REPOS else r for r in ALL_REPOS],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_yticklabels([bucket_label(b) for b in ACTIVE_BUCKETS],
                       rotation=0, fontsize=9)
    ax.set_ylabel("Difficulty Bucket")
    ax.set_title("Injector Coverage by Difficulty Bucket × Repository\n"
                 "(ok cells / total enumerated, all sources × types)",
                 fontsize=12, pad=12)
    ax.axhline(y=9, color="black", linewidth=1.5)
    ax.axvline(x=len(TRAIN_REPOS), color="black", linewidth=1.5)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved coverage heatmap to {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="P1.24 difficulty-bucket heatmap")
    parser.add_argument("--input", default=None,
                        help="baseline_eval.json with pass@1 data")
    parser.add_argument("--coverage", action="store_true",
                        help="generate injector coverage heatmap (no model needed)")
    parser.add_argument("--out", default=None,
                        help="output path (default: artifacts/p1/base_model_heatmap.png)")
    args = parser.parse_args()

    if args.coverage:
        out = Path(args.out) if args.out else ROOT / "artifacts/p1/injector_coverage_heatmap.png"
        heatmap_from_coverage(out)
    elif args.input:
        data = json.loads(Path(args.input).read_text())
        out = Path(args.out) if args.out else ROOT / "artifacts/p1/base_model_heatmap.png"
        heatmap_from_eval(data, out)
    else:
        print("Specify --input baseline_eval.json or --coverage", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
