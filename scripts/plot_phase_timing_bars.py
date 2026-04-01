#!/usr/bin/env python3
"""Bar chart: mean phase time (ms) vs batch size from phase_timing summary CSVs.

Scans a directory tree for files matching ``*_phase_timing_*_summary.csv`` (as
written by ``src/trainer/stats/phase_timing.py``). Parent folders are expected
to be named ``batch_<N>_worker_<M>`` (same convention as Whisper sweeps).

Aggregation rule (documented):
  For each batch size *N*, all summary files under ``batch_N_worker_*`` are
  collected. Summary CSVs use **milliseconds** (``mean_*_ms``). Legacy summaries
  with ``mean_*_s`` are converted ×1000 for the plot. Bar height is the **mean of
  per-run means** (ms). Error bars are the **standard deviation of those per-run
  means** across runs in the group. If only one run exists for a batch, the
  error bar is zero.

Usage:
  python3 scripts/plot_phase_timing_bars.py results/data
  python3 scripts/plot_phase_timing_bars.py results/data --out results/plots/phase_timing_by_batch.png
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BATCH_DIR_RE = re.compile(r"batch_(\d+)_worker_(\d+)")
SUMMARY_GLOB = "*_phase_timing_*_summary.csv"

PHASE_SPEC = (
    ("forward", "Forward", "#2c7bb6"),
    ("backward", "Backward", "#fdae61"),
    ("optimizer", "Optimizer", "#7fbc41"),
)


def _mean_ms_from_row(row: pd.Series, phase: str) -> float:
    """Read mean time for ``phase`` from a summary row (ms, or legacy seconds)."""
    k_ms = f"mean_{phase}_ms"
    k_s = f"mean_{phase}_s"
    if k_ms in row.index and pd.notna(row[k_ms]):
        return float(row[k_ms])
    if k_s in row.index and pd.notna(row[k_s]):
        return float(row[k_s]) * 1000.0
    return float("nan")


def discover_summaries(root: Path) -> list[tuple[Path, int, int]]:
    """Return (csv_path, batch_size, worker_id) for each summary under root."""
    out: list[tuple[Path, int, int]] = []
    for path in root.rglob(SUMMARY_GLOB):
        if not path.is_file():
            continue
        parent_name = path.resolve().parent.name
        m = BATCH_DIR_RE.fullmatch(parent_name)
        if not m:
            continue
        out.append((path, int(m.group(1)), int(m.group(2))))
    return sorted(out, key=lambda t: (t[1], t[2], str(t[0])))


def aggregate_by_batch(
    rows: list[tuple[Path, int, int]],
) -> dict[int, dict[str, np.ndarray]]:
    """batch_size -> arrays of shape (n_runs,) for each mean_* column."""
    by_batch: dict[int, list[pd.Series]] = defaultdict(list)
    for csv_path, bsz, _worker in rows:
        df = pd.read_csv(csv_path)
        if len(df) != 1:
            raise SystemExit(f"{csv_path}: expected exactly one summary row, got {len(df)}")
        by_batch[bsz].append(df.iloc[0])
    stats: dict[int, dict[str, np.ndarray]] = {}
    for bsz, series_list in sorted(by_batch.items()):
        stats[bsz] = {
            phase: np.array([_mean_ms_from_row(s, phase) for s in series_list])
            for phase, _label, _c in PHASE_SPEC
        }
    return stats


def plot_grouped_bars(
    stats_by_batch: dict[int, dict[str, np.ndarray]],
    out_path: Path,
    *,
    title: str,
) -> None:
    batches = sorted(stats_by_batch.keys())
    if not batches:
        raise SystemExit("No batch sizes to plot (no matching summary CSVs).")

    x = np.arange(len(batches), dtype=float)
    n_phases = len(PHASE_SPEC)
    width = min(0.22, 0.8 / (n_phases + 1))

    fig, ax = plt.subplots(figsize=(max(5.0, 1.2 * len(batches)), 5.5), layout="constrained")
    for i, (phase_key, label, color) in enumerate(PHASE_SPEC):
        means = np.array([np.nanmean(stats_by_batch[b][phase_key]) for b in batches])
        errs = np.array(
            [np.nanstd(stats_by_batch[b][phase_key], ddof=0) for b in batches]
        )
        pos = x + (i - (n_phases - 1) / 2.0) * width
        ax.bar(
            pos,
            means,
            width * 0.92,
            yerr=errs,
            label=label,
            color=color,
            capsize=3,
            alpha=0.92,
        )

    ax.set_xticks(x, [str(b) for b in batches])
    ax.set_xlabel("Batch size", fontsize=11)
    ax.set_ylabel("Mean time (ms)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(title="Phase", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root",
        type=Path,
        help="Directory to scan (e.g. results/data)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/plots/phase_timing_by_batch.png"),
        help="Output PNG path",
    )
    p.add_argument(
        "--title",
        default="Phase timing — mean ± spread across runs (ms)",
        help="Figure title",
    )
    args = p.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    found = discover_summaries(root)
    if not found:
        raise SystemExit(
            f"No files matching {SUMMARY_GLOB!r} under {root} "
            "with parent dir batch_<N>_worker_<M>."
        )

    stats = aggregate_by_batch(found)
    plot_grouped_bars(stats, args.out.resolve(), title=args.title)
    print(f"Used {len(found)} summary file(s) across batch sizes {sorted(stats.keys())}")


if __name__ == "__main__":
    main()
