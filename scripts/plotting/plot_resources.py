#!/usr/bin/env python3
"""
Plot resource utilization from resource_util.csv.

Produces two plots when phase data is available:
  1. resource_util.png - overview (aggregated by step, no phase breakdown)
  2. resource_util_phases.png - boxplots grouped by phase (forward/backward/optimizer)

Usage:
    python scripts/plotting/plot_resources.py [--input PATH] [--output-dir DIR]
    python scripts/plotting/plot_resources.py --zoom 10 [--zoom-start 1]  # zoomed view of N steps
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


METRICS_OVERVIEW = [
    ("gpu_util", "GPU Util (%)", "GPU Utilization"),
    ("cpu_util", "CPU Util (%)", "CPU Utilization"),
    ("gpu_mem_pct", "%", "GPU Memory (consumed/allocated)"),
    ("cpu_mem_gb", "GB", "CPU Memory"),
    ("ram_gb", "GB", "System RAM"),
    ("io_read_gb", "GB", "I/O Read"),
    ("io_write_gb", "GB", "I/O Write"),
]

# Metrics where phase breakdown is meaningful (excludes cumulative I/O and near-constant RAM)
METRICS_BOXPLOT = [
    ("gpu_util", "GPU Util (%)", "GPU Utilization"),
    ("cpu_util", "CPU Util (%)", "CPU Utilization"),
    ("gpu_mem_pct", "%", "GPU Memory (consumed/allocated)"),
    ("cpu_mem_gb", "GB", "CPU Memory"),
]


def _setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass


def plot_overview(df: pd.DataFrame, output_path: Path) -> None:
    """Line plot aggregated by step. Uses 2x4 grid for compact layout."""
    _setup_style()
    has_phase = "phase" in df.columns
    if has_phase:
        df_plot = df.groupby("step", as_index=False).mean(numeric_only=True)
    else:
        df_plot = df

    metrics = METRICS_OVERVIEW
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    axes = axes.flatten()

    for idx, (col, ylabel, title) in enumerate(metrics):
        ax = axes[idx]
        if col not in df_plot.columns:
            ax.text(0.5, 0.5, f"Column '{col}' not found", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        ax.plot(df_plot["step"], df_plot[col], linewidth=1.2, color="#2980b9")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="medium")
        ymax = df_plot[col].max()
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.grid(True, alpha=0.3)
        if col == "io_write_gb" and df_plot[col].max() < 1e-3:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e6:.0f}"))
            ax.set_ylabel("KB", fontsize=9)
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")
    fig.supxlabel("Step", fontsize=10, y=-0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved overview to {output_path}")


def plot_gpu_cpu_overlap(df: pd.DataFrame, output_path: Path, cpu_cores: Optional[int] = None) -> None:
    """GPU vs CPU utilization overlaid for direct comparison.

    When phase data exists, creates 3 subplots (forward/backward/optimizer).
    When cpu_cores is set, CPU util is normalized so both metrics are 0-100%.
    """
    _setup_style()
    if "gpu_util" not in df.columns or "cpu_util" not in df.columns:
        print("Skipping GPU/CPU overlap plot: gpu_util or cpu_util not in data")
        return

    phase_order = ["forward", "backward", "optimizer"]
    has_phase = "phase" in df.columns and df["phase"].isin(phase_order).any()

    def _plot_overlay(ax: plt.Axes, df_plot: pd.DataFrame, title: str) -> None:
        cpu_util = df_plot["cpu_util"].copy()
        if cpu_cores is not None and cpu_cores > 0:
            cpu_util = cpu_util / cpu_cores
        ax.plot(df_plot["step"], df_plot["gpu_util"], linewidth=1.5, color="#2980b9", label="GPU Util (%)")
        ax.plot(df_plot["step"], cpu_util, linewidth=1.5, color="#e74c3c",
                label="CPU Util (%)" + (f" (norm/{cpu_cores})" if cpu_cores else ""))
        ax.set_ylabel("Utilization (%)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="medium")
        ax.legend(loc="upper right", fontsize=8)
        ymax = max(df_plot["gpu_util"].max(), cpu_util.max())
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.grid(True, alpha=0.3)

    if has_phase:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax, phase in zip(axes, phase_order):
            df_phase = df[df["phase"] == phase].copy()
            if df_phase.empty:
                ax.text(0.5, 0.5, f"No data for {phase}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(phase.capitalize(), fontsize=10, fontweight="medium")
                continue
            df_phase = df_phase.sort_values("step")
            _plot_overlay(ax, df_phase, phase.capitalize())
        axes[-1].set_xlabel("Step", fontsize=10)
    else:
        df_plot = df.groupby("step", as_index=False).mean(numeric_only=True) if "step" in df.columns else df
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_overlay(ax, df_plot, "GPU vs CPU Utilization")
        ax.set_xlabel("Step", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved GPU/CPU overlap to {output_path}")


def plot_phases_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    """Violin plot of each metric by phase. 2x2 grid, shows full distribution shape."""
    _setup_style()
    phase_order = ["forward", "backward", "optimizer"]
    phase_colors = {"forward": "#27ae60", "backward": "#e74c3c", "optimizer": "#3498db"}
    df_plot = df[df["phase"].isin(phase_order)].copy()

    metrics = METRICS_BOXPLOT
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (col, ylabel, title) in zip(axes, metrics):
        if col not in df_plot.columns:
            ax.text(0.5, 0.5, f"Column '{col}' not found", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        data = [df_plot.loc[df_plot["phase"] == p, col].dropna().values for p in phase_order]
        parts = ax.violinplot(data, positions=[0, 1, 2], widths=0.7, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(phase_colors[phase_order[i]])
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")
            pc.set_linewidth(1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(phase_order, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="medium")
        ax.set_ylabel(ylabel, fontsize=9)
        ymax = max((np.max(d) for d in data if len(d) > 0), default=1)
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
        ax.grid(True, alpha=0.3, axis="y")
    fig.supxlabel("Phase", fontsize=10, y=-0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved phase plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot resource utilization from CSV")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "logs" / "resource_util.csv",
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: scripts/plotting/)",
    )
    parser.add_argument(
        "--zoom", "-z",
        type=int,
        default=None,
        metavar="N",
        help="Also produce a zoomed overview of N steps (e.g. --zoom 10)",
    )
    parser.add_argument(
        "--zoom-start",
        type=int,
        default=1,
        metavar="STEP",
        help="First step for zoom window (default: 1)",
    )
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=4,
        metavar="N",
        help="Normalize CPU util by N cores for GPU/CPU overlap (0-100%% scale). Default: 4 (SLURM)",
    )
    parser.add_argument(
        "--no-normalize-cpu",
        action="store_true",
        help="Disable CPU normalization in overlap plot (show raw %% per core)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    has_phase = "phase" in df.columns

    output_dir = args.output_dir or Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu_cores = None if args.no_normalize_cpu else args.cpu_cores

    plot_overview(df, output_dir / "resource_util.png")
    plot_gpu_cpu_overlap(df, output_dir / "resource_util_gpu_cpu.png", cpu_cores=cpu_cores)

    if has_phase:
        plot_phases_boxplot(df, output_dir / "resource_util_phases.png")

    if args.zoom is not None:
        step_end = args.zoom_start + args.zoom - 1
        mask = (df["step"] >= args.zoom_start) & (df["step"] <= step_end)
        df_zoom = df[mask]
        if df_zoom.empty:
            raise ValueError(f"No data in step range {args.zoom_start}-{step_end}")
        zoom_path = output_dir / f"resource_util_zoom_steps_{args.zoom_start}-{step_end}.png"
        plot_overview(df_zoom, zoom_path)
        plot_gpu_cpu_overlap(df_zoom, output_dir / f"resource_util_gpu_cpu_zoom_steps_{args.zoom_start}-{step_end}.png", cpu_cores=cpu_cores)


if __name__ == "__main__":
    main()
