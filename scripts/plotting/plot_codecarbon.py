#!/usr/bin/env python3
"""
Plot CodeCarbon emissions and energy data from CSV.

Handles both:
  - Full-run summary (single row: emissions, energy by component, power, duration)
  - Step/substep CSVs (multiple rows with task-level data)

Usage:
    python scripts/plotting/plot_codecarbon.py
    python scripts/plotting/plot_codecarbon.py --zoom 50
    python scripts/plotting/plot_codecarbon.py --smooth 10  # rolling average (default: 5)

Reads from logs/ (*-steps.csv, *-substeps.csv, or code_carbon_base.csv).
Writes to scripts/plotting/.
"""
import argparse
import re
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise ImportError("matplotlib and numpy required. Install with: pip install matplotlib numpy")

# Phase mapping from task_name prefix to canonical phase
PHASE_MAP = {
    "Forward pass": "forward",
    "Backward pass": "backward",
    "Optimisation step": "optimizer",
}
PHASE_ORDER = ["forward", "backward", "optimizer"]
PHASE_COLORS = {"forward": "#27ae60", "backward": "#e74c3c", "optimizer": "#3498db"}


def _parse_step_num(task_name: str) -> Optional[int]:
    """Extract step number from task_name (e.g. 'Step #5' or 'Forward pass #5' -> 5)."""
    m = re.search(r"#(\d+)", str(task_name))
    return int(m.group(1)) if m else None


def _parse_phase(task_name: str) -> Optional[str]:
    """Extract phase from task_name (e.g. 'Forward pass #5' -> 'forward')."""
    s = str(task_name)
    for prefix, phase in PHASE_MAP.items():
        if s.startswith(prefix):
            return phase
    return None


def _detect_csv_type(df: pd.DataFrame) -> Literal["step", "substep", "generic"]:
    """Infer CSV type from task_name patterns."""
    if "task_name" not in df.columns or df.empty:
        return "generic"
    names = df["task_name"].astype(str)
    step_pattern = names.str.match(r"^Step #\d+$", na=False)
    substep_pattern = names.str.match(
        r"^(Forward pass|Backward pass|Optimisation step) #\d+$", na=False
    )
    if step_pattern.all():
        return "step"
    if substep_pattern.all():
        return "substep"
    return "generic"


def _apply_zoom(df: pd.DataFrame, zoom: Optional[int], zoom_start: int) -> pd.DataFrame:
    """Filter df to steps in [zoom_start, zoom_start + zoom - 1] if zoom is set."""
    if zoom is None or "step" not in df.columns:
        return df
    step_end = zoom_start + zoom - 1
    mask = (df["step"] >= zoom_start) & (df["step"] <= step_end)
    return df[mask].copy()


def _setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass


def _smooth_series(series: pd.Series, window: int) -> pd.Series:
    """Apply centered rolling mean. No-op if window <= 1."""
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def plot_full_run_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Dashboard for a single-run CodeCarbon summary (one row)."""
    _setup_style()
    row = df.iloc[0]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # 1. Energy breakdown (bar)
    ax1 = fig.add_subplot(gs[0, 0])
    energy_cols = ["cpu_energy", "gpu_energy", "ram_energy"]
    labels = ["CPU", "GPU", "RAM"]
    vals = [row.get(c, 0) for c in energy_cols]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax1.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Energy (kWh)", fontsize=10)
    ax1.set_title("Energy by Component", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # 2. Power breakdown (bar)
    ax2 = fig.add_subplot(gs[0, 1])
    power_cols = ["cpu_power", "gpu_power", "ram_power"]
    labels = ["CPU", "GPU", "RAM"]
    vals = [row.get(c, 0) for c in power_cols]
    bars = ax2.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Power (W)", fontsize=10)
    ax2.set_title("Power by Component", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # 3. Summary metrics (text)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    duration = row.get("duration", 0)
    emissions = row.get("emissions", 0)
    energy = row.get("energy_consumed", 0)
    project = row.get("project_name", "N/A")
    gpu = row.get("gpu_model", "N/A")
    region = row.get("region", row.get("country_name", "N/A"))

    text = (
        f"Project: {project}  |  Region: {region}  |  GPU: {gpu}\n\n"
        f"Duration: {duration:.1f} s ({duration/60:.2f} min)\n"
        f"Total energy: {energy:.5f} kWh\n"
        f"Emissions: {emissions:.6f} kg CO₂eq"
    )
    ax3.text(0.5, 0.5, text, transform=ax3.transAxes,
            fontsize=11, va="center", ha="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            family="monospace")

    fig.suptitle("CodeCarbon Run Summary", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary to {output_path}")


def plot_step_data(
    df: pd.DataFrame,
    output_path: Path,
    zoom: Optional[int] = None,
    zoom_start: int = 1,
    smooth: int = 1,
) -> None:
    """Line plots for step/substep-level CodeCarbon data (multiple rows)."""
    _setup_style()

    # Parse step number from task_name if present
    if "task_name" in df.columns:
        df = df.copy()
        df["step"] = df["task_name"].apply(_parse_step_num)
        df = df.dropna(subset=["step"])
        df["step"] = df["step"].astype(int)
        if "timestamp" in df.columns:
            df = df.sort_values(["step", "timestamp"]).reset_index(drop=True)
        else:
            df = df.sort_values("step").reset_index(drop=True)

    df = _apply_zoom(df, zoom, zoom_start)
    if df.empty:
        print("No data after zoom filter, skipping step plot")
        return

    # Use step number on x-axis when available, else row index
    if "step" in df.columns and df["step"].notna().all():
        x = df["step"].values
        xlabel = "Step"
    else:
        x = np.arange(len(df))
        xlabel = "Task index"

    metrics = [
        ("emissions", "Emissions (kg CO₂eq)", "#e74c3c"),
        ("energy_consumed", "Energy (kWh)", "#3498db"),
        ("duration", "Duration (s)", "#2ecc71"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, (col, ylabel, color) in zip(axes, metrics):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"Column '{col}' not found", ha="center", va="center", transform=ax.transAxes)
            continue
        y = _smooth_series(df[col], smooth)
        ax.plot(x, y, linewidth=1.2, color=color)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="medium")
        ax.grid(True, alpha=0.3)
        ymax = y.max()
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)

    axes[-1].set_xlabel(xlabel, fontsize=10)
    fig.suptitle("CodeCarbon Step-level Metrics", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved step plot to {output_path}")


def _prepare_substep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse phase and step from task_name, return df with phase and step columns."""
    out = df.copy()
    out["step"] = out["task_name"].apply(_parse_step_num)
    out["phase"] = out["task_name"].apply(_parse_phase)
    out = out.dropna(subset=["step", "phase"])
    out["step"] = out["step"].astype(int)
    return out


def plot_phases_boxplot(
    df: pd.DataFrame,
    output_path: Path,
    zoom: Optional[int] = None,
    zoom_start: int = 1,
) -> None:
    """Boxplot of emissions distribution by phase (forward/backward/optimizer)."""
    _setup_style()
    df_plot = _prepare_substep_df(df)
    df_plot = _apply_zoom(df_plot, zoom, zoom_start)
    if df_plot.empty or "emissions" not in df_plot.columns:
        print("Skipping phases boxplot: no emissions or empty data")
        return

    df_plot = df_plot[df_plot["phase"].isin(PHASE_ORDER)]
    data = [df_plot.loc[df_plot["phase"] == p, "emissions"].dropna().values for p in PHASE_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.boxplot(data, positions=[0, 1, 2], widths=0.6, patch_artist=True)
    for i, patch in enumerate(parts["boxes"]):
        patch.set_facecolor(PHASE_COLORS[PHASE_ORDER[i]])
        patch.set_alpha(0.7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(PHASE_ORDER, fontsize=10)
    ax.set_ylabel("Emissions (kg CO₂eq)", fontsize=10)
    ax.set_title("Emissions Distribution by Phase", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ymax = max((np.max(d) for d in data if len(d) > 0), default=1)
    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved phases boxplot to {output_path}")


def plot_phases_stacked(
    df: pd.DataFrame,
    output_path: Path,
    zoom: Optional[int] = None,
    zoom_start: int = 1,
) -> None:
    """Stacked bar chart: emissions by phase for each step."""
    _setup_style()
    df_plot = _prepare_substep_df(df)
    df_plot = _apply_zoom(df_plot, zoom, zoom_start)
    if df_plot.empty or "emissions" not in df_plot.columns:
        print("Skipping phases stacked: no emissions or empty data")
        return

    pivot = df_plot.pivot_table(
        index="step", columns="phase", values="emissions", aggfunc="sum"
    )
    for p in PHASE_ORDER:
        if p not in pivot.columns:
            pivot[p] = 0
    pivot = pivot[[p for p in PHASE_ORDER if p in pivot.columns]]

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(pivot))
    for i, phase in enumerate(PHASE_ORDER):
        if phase in pivot.columns:
            ax.bar(pivot.index, pivot[phase], bottom=bottom, label=phase.capitalize(),
                   color=PHASE_COLORS[phase], edgecolor="black", linewidth=0.5)
            bottom += pivot[phase].values

    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Emissions (kg CO₂eq)", fontsize=10)
    ax.set_title("Emissions by Phase per Step", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved phases stacked to {output_path}")


def plot_phases_pie(
    df: pd.DataFrame,
    output_path: Path,
    zoom: Optional[int] = None,
    zoom_start: int = 1,
) -> None:
    """Pie chart: total emissions share by phase."""
    _setup_style()
    df_plot = _prepare_substep_df(df)
    df_plot = _apply_zoom(df_plot, zoom, zoom_start)
    if df_plot.empty or "emissions" not in df_plot.columns:
        print("Skipping phases pie: no emissions or empty data")
        return

    by_phase = df_plot.groupby("phase")["emissions"].sum()
    by_phase = by_phase.reindex(PHASE_ORDER).fillna(0)
    labels = [p.capitalize() for p in PHASE_ORDER]
    colors = [PHASE_COLORS[p] for p in PHASE_ORDER]
    sizes = by_phase.values

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
        explode=(0.02,) * len(sizes)
    )
    for w in wedges:
        w.set_edgecolor("black")
        w.set_linewidth(0.5)
    ax.set_title("Total Emissions by Phase", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved phases pie to {output_path}")


def plot_energy_by_phase(
    df: pd.DataFrame,
    output_path: Path,
    zoom: Optional[int] = None,
    zoom_start: int = 1,
) -> None:
    """Stacked bar: CPU/GPU/RAM energy per phase (forward, backward, optimizer)."""
    _setup_style()
    df_plot = _prepare_substep_df(df)
    df_plot = _apply_zoom(df_plot, zoom, zoom_start)
    energy_cols = ["cpu_energy", "gpu_energy", "ram_energy"]
    if df_plot.empty or not any(c in df_plot.columns for c in energy_cols):
        print("Skipping energy by phase: no energy columns or empty data")
        return

    for c in energy_cols:
        if c not in df_plot.columns:
            df_plot[c] = 0

    by_phase = df_plot.groupby("phase")[energy_cols].sum()
    by_phase = by_phase.reindex(PHASE_ORDER).fillna(0)

    x = np.arange(len(PHASE_ORDER))
    width = 0.6
    comp_colors = ["#3498db", "#e74c3c", "#2ecc71"]
    comp_labels = ["CPU", "GPU", "RAM"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(PHASE_ORDER))
    for i, col in enumerate(energy_cols):
        vals = by_phase[col].values if col in by_phase.columns else np.zeros(len(PHASE_ORDER))
        ax.bar(x, vals, width, bottom=bottom, label=comp_labels[i], color=comp_colors[i],
               edgecolor="black", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PHASE_ORDER])
    ax.set_ylabel("Energy (kWh)", fontsize=10)
    ax.set_title("Energy by Component and Phase", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved energy by phase to {output_path}")


def _aggregate_summary_row(df: pd.DataFrame) -> dict:
    """Build a single-row summary dict from multi-row task data."""
    return {
        "duration": df["duration"].sum() if "duration" in df.columns else 0,
        "emissions": df["emissions"].sum(),
        "energy_consumed": df["energy_consumed"].sum(),
        "cpu_energy": df["cpu_energy"].sum() if "cpu_energy" in df.columns else 0,
        "gpu_energy": df["gpu_energy"].sum() if "gpu_energy" in df.columns else 0,
        "ram_energy": df["ram_energy"].sum() if "ram_energy" in df.columns else 0,
        "cpu_power": df["cpu_power"].mean() if "cpu_power" in df.columns else 0,
        "gpu_power": df["gpu_power"].mean() if "gpu_power" in df.columns else 0,
        "ram_power": df["ram_power"].mean() if "ram_power" in df.columns else 0,
        "project_name": df["project_name"].iloc[0] if "project_name" in df.columns else "N/A",
        "gpu_model": df["gpu_model"].iloc[0] if "gpu_model" in df.columns else "N/A",
        "region": df["region"].iloc[0] if "region" in df.columns else "N/A",
        "country_name": df["country_name"].iloc[0] if "country_name" in df.columns else "N/A",
    }


def _process_file(
    input_path: Path,
    output_dir: Path,
    zoom: Optional[int],
    zoom_start: int,
    smooth: int = 1,
) -> None:
    """Process a single CodeCarbon CSV and produce appropriate plots."""
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("CSV is empty")

    base_name = input_path.stem
    csv_type = _detect_csv_type(df)

    if len(df) == 1:
        plot_full_run_summary(df, output_dir / f"{base_name}_summary.png")
        return

    if csv_type == "step":
        plot_step_data(df, output_dir / f"{base_name}_steps.png", zoom=None, zoom_start=zoom_start, smooth=smooth)
        if zoom is not None:
            zoom_end = zoom_start + zoom - 1
            plot_step_data(
                df, output_dir / f"{base_name}_steps_zoom_{zoom_start}-{zoom_end}.png",
                zoom=zoom, zoom_start=zoom_start, smooth=smooth
            )
    elif csv_type == "substep":
        plot_phases_boxplot(df, output_dir / f"{base_name}_phases.png", zoom=None, zoom_start=zoom_start)
        plot_phases_stacked(df, output_dir / f"{base_name}_phases_stacked.png", zoom=None, zoom_start=zoom_start)
        plot_phases_pie(df, output_dir / f"{base_name}_phases_pie.png", zoom=None, zoom_start=zoom_start)
        plot_energy_by_phase(df, output_dir / f"{base_name}_energy_by_phase.png", zoom=None, zoom_start=zoom_start)
        if zoom is not None:
            zoom_end = zoom_start + zoom - 1
            suffix = f"_zoom_{zoom_start}-{zoom_end}"
            plot_phases_boxplot(df, output_dir / f"{base_name}_phases{suffix}.png", zoom=zoom, zoom_start=zoom_start)
            plot_phases_stacked(df, output_dir / f"{base_name}_phases_stacked{suffix}.png", zoom=zoom, zoom_start=zoom_start)
            plot_phases_pie(df, output_dir / f"{base_name}_phases_pie{suffix}.png", zoom=zoom, zoom_start=zoom_start)
            plot_energy_by_phase(df, output_dir / f"{base_name}_energy_by_phase{suffix}.png", zoom=zoom, zoom_start=zoom_start)
    else:
        plot_step_data(df, output_dir / f"{base_name}_steps.png", zoom=zoom, zoom_start=zoom_start, smooth=smooth)

    if "emissions" in df.columns and "energy_consumed" in df.columns:
        plot_full_run_summary(
            pd.DataFrame([_aggregate_summary_row(df)]),
            output_dir / f"{base_name}_summary.png",
        )


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    input_dir = project_root / "logs"
    output_dir = script_dir

    parser = argparse.ArgumentParser(description="Plot CodeCarbon emissions/energy from CSV")
    parser.add_argument(
        "--zoom", "-z",
        type=int,
        default=None,
        metavar="N",
        help="Limit plots to first N steps (e.g. --zoom 50)",
    )
    parser.add_argument(
        "--zoom-start",
        type=int,
        default=1,
        metavar="STEP",
        help="First step for zoom window (default: 1)",
    )
    parser.add_argument(
        "--smooth", "-s",
        type=int,
        default=5,
        metavar="N",
        help="Rolling window size for line smoothing (default: 5). Use 1 to disable.",
    )
    args = parser.parse_args()

    output_dir.mkdir(parents=True, exist_ok=True)

    step_files = sorted(input_dir.glob("*-steps.csv"))
    substep_files = sorted(input_dir.glob("*-substeps.csv"))
    files = list(step_files) + list(substep_files)
    if not files:
        fallback = input_dir / "code_carbon_base.csv"
        if fallback.exists():
            files = [fallback]
        else:
            raise FileNotFoundError(f"No *-steps.csv, *-substeps.csv, or code_carbon_base.csv in {input_dir}")

    for f in files:
        _process_file(f, output_dir, args.zoom, args.zoom_start, args.smooth)


if __name__ == "__main__":
    main()
