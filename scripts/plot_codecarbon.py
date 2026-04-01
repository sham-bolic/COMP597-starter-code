#!/usr/bin/env python3
"""Plot CodeCarbon CSV logs produced by ``src/trainer/stats/codecarbon.py``.

File layout (same directory):

- ``run_N_cc_full_rank_R.csv`` — one row per flush: whole-training summary.
- ``run_N_cc_step_rank_R.csv`` — tracker summary; per-step tasks live in
  ``run_N_cc_step_rank_R-steps.csv``.
- ``run_N_cc_substep_rank_R.csv`` — substep tracker summary; task rows in
  ``run_N_cc_substep_rank_R-substeps.csv``.

Examples:

  python3 scripts/plot_codecarbon.py results/data/batch_4_worker_0/run_1_cc_step_rank_None-steps.csv
  python3 scripts/plot_codecarbon.py results/data/batch_4_worker_0/run_1_cc_substep_rank_None-substeps.csv
  python3 scripts/plot_codecarbon.py results/data/batch_4_worker_0
  python3 scripts/plot_codecarbon.py results/data/batch_4_worker_0/run_1_cc_full_rank_None.csv

Passing a **directory** groups ``run_*`` step/substep task CSVs, averages across runs, and writes
outputs under ``results/plots/<mirrored-path>/``. ``cc_full`` files are **not** plotted in batch
mode. ``cc_full`` is plotted only for a **single** CSV with **two or more rows**.

**Per-step figures** (metrics, hardware, subphases, duration PNG) drop the first
``CODECARBON_PLOT_WARMUP_STEPS`` training steps by sorted index and tighten the x-axis; cumulative
series sum only over the plotted tail. Summary CSVs still use all steps unless you trim elsewhere.
Adjust the constant in this file to change warm-up length.

Y-axes use **scientific notation** (e.g. ``1.2e-04``) for **energy and emissions** only; duration
and other quantities keep default formatting. Step-index x-axes stay plain integers.

**Outputs this script creates** — let ``OUT`` = ``results/plots/<mirrored-path>/`` (mirror of the
path under ``results/`` for the input file or directory). Template stems omit ``run_<n>_``:
``<step-stem>`` ≈ ``cc_step_rank_None-steps``, ``<sub-stem>`` ≈ ``cc_substep_rank_None-substeps``.

*Directory batch*

- ``OUT/<step-stem>_mean_codecarbon_metrics.png``
- ``OUT/<step-stem>_mean_codecarbon_hardware_energy.png``
- ``OUT/<sub-stem>_mean_codecarbon_subphase_energy.png``
- ``OUT/<sub-stem>_mean_codecarbon_subphase_energy_isolated.png``
- ``OUT/<sub-stem>_mean_codecarbon_total_energy.png``
- ``OUT/<step-stem>_codecarbon_duration_by_step.csv`` / ``_summary.csv`` / ``_duration.png`` when
  matching substep logs exist
- ``OUT/codecarbon_summary_table.csv`` (includes ``cc_full`` rows found in that directory)

*Single file* — ``STEM`` is the stem of the task CSV (e.g. ``run_1_cc_step_rank_None-steps``):

- **Steps:** ``OUT/{STEM}_codecarbon_metrics.png``, ``OUT/{STEM}_codecarbon_hardware_energy.png``,
  ``OUT/{STEM}_codecarbon_summary_table.csv``, and duration CSVs + ``_codecarbon_duration.png`` if a
  paired substep file exists
- **Substeps:** ``OUT/{STEM}_codecarbon_subphase_energy.png``,
  ``OUT/{STEM}_codecarbon_subphase_energy_isolated.png``,
  ``OUT/{STEM}_codecarbon_subphase_energy_total_energy.png``,
  ``OUT/{STEM}_codecarbon_summary_table.csv``, plus duration CSVs + ``_codecarbon_duration.png`` if a paired
  step file exists (artifacts use the substep ``STEM`` in that case)
- **Full:** ``OUT/{STEM}_codecarbon_summary_table.csv`` always; ``OUT/{STEM}_codecarbon_summary.png``
  only if the CSV has ≥2 rows

**Duration tables** (when both step and substep logs exist / match): ``{STEM}_codecarbon_duration_by_step.csv``
(per-step whole step vs forward/backward/optimizer seconds), ``{STEM}_codecarbon_duration_summary.csv``
(mean/median/std/min/max), and ``{STEM}codecarbon_duration.png`` (whole vs subphase lines, stacked
phase bars, per-metric means in **ms** after the same warm-up trim; CSVs stay in **seconds**).
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd

RESULTS_DIR_NAME = "results"
DEFAULT_PLOTS_ROOT = Path("results") / "plots"

STEP_TASK_RE = re.compile(r"^Step #(\d+)\s*$", re.IGNORECASE)
SUBSTEP_TASK_RE = re.compile(
    r"^(?P<phase>Forward pass|Backward pass|Optimisation step)\s*#(?P<num>\d+)\s*$",
    re.IGNORECASE,
)
RUN_PREFIX_RE = re.compile(r"^run_(\d+)_", re.IGNORECASE)


def _run_sort_key(path: Path) -> tuple[int, str]:
    m = RUN_PREFIX_RE.match(path.name)
    n = int(m.group(1)) if m else 0
    return (n, path.name)


def _strip_run_prefix(filename: str) -> tuple[int | None, str]:
    """``run_1_cc_full_x.csv`` → ``(1, 'cc_full_x.csv')``; no prefix → ``(None, filename)``."""
    m = RUN_PREFIX_RE.match(filename)
    if not m:
        return None, filename
    return int(m.group(1)), filename[m.end() :]


def _template_key(path: Path) -> str:
    """Filename without ``run_<n>_`` so run 1/2/3 files group together."""
    _, rest = _strip_run_prefix(path.name)
    return rest


def _paired_substep_template_key(step_template_key: str) -> str | None:
    """``cc_step_...-steps.csv`` → ``cc_substep_...-substeps.csv`` template key if pairing applies."""
    if "cc_step_" not in step_template_key or not step_template_key.endswith("-steps.csv"):
        return None
    return step_template_key.replace("cc_step_", "cc_substep_").replace("-steps.csv", "-substeps.csv")


def _mirror_plots_subdir(ref_path: Path, plots_root: Path) -> Path:
    ref_path = ref_path.resolve()
    parts = ref_path.parts
    try:
        i = parts.index(RESULTS_DIR_NAME)
    except ValueError:
        rel = Path(ref_path.parent.name)
    else:
        inner = parts[i + 1 : -1]
        rel = Path(*inner) if inner else Path()
    return plots_root.resolve() / rel


def _parse_step_index(task_name: str) -> int | None:
    m = STEP_TASK_RE.match(str(task_name).strip())
    return int(m.group(1)) if m else None


def _parse_substep(task_name: str) -> tuple[str, int] | None:
    m = SUBSTEP_TASK_RE.match(str(task_name).strip())
    if not m:
        return None
    phase = m.group("phase").lower().replace(" ", "_")
    return phase, int(m.group("num"))


def _xlim_from_zero(ax, x: pd.Series, *, pad_ratio: float = 0.02) -> None:
    xs = pd.to_numeric(x, errors="coerce")
    if not xs.notna().any():
        ax.set_xlim(0, 1)
        return
    xmax = float(xs.max(skipna=True))
    ax.set_xlim(0, max(xmax * (1.0 + pad_ratio), 1.0))


# Per-step CodeCarbon plots: drop the first N rows by sorted ``x_col`` so y-axes ignore warm-up spikes.
CODECARBON_PLOT_WARMUP_STEPS = 10


def _trim_codecarbon_warmup(df: pd.DataFrame, x_col: str) -> tuple[pd.DataFrame, int]:
    if x_col not in df.columns or len(df) == 0:
        return df, 0
    out = df.sort_values(x_col, kind="mergesort").reset_index(drop=True)
    n_skip = min(CODECARBON_PLOT_WARMUP_STEPS, max(0, len(out) - 1))
    if n_skip:
        out = out.iloc[n_skip:].reset_index(drop=True)
    return out, n_skip


def _append_warmup_to_caption(caption: str, n_skip: int, *, cumulative_note: bool = False) -> str:
    if not n_skip:
        return caption
    extra = f" First {n_skip} step(s) by index omitted (warm-up trim)."
    if cumulative_note:
        extra += " Cumulative curves sum only over plotted steps."
    return caption + extra


def _xlim_after_warmup(ax, x: pd.Series, n_skip: int, *, pad_ratio: float = 0.02) -> None:
    xs = pd.to_numeric(x, errors="coerce")
    if n_skip and xs.notna().any():
        xa = float(xs.min(skipna=True))
        xb = float(xs.max(skipna=True))
        span = max(xb - xa, 1.0)
        ax.set_xlim(max(0.0, xa - pad_ratio * span), xb + pad_ratio * span)
    else:
        _xlim_from_zero(ax, x)


def _ylim_from_zero(ax, y, *, pad_ratio: float = 0.06) -> None:
    vals = pd.to_numeric(pd.Series(y), errors="coerce")
    if not vals.notna().any():
        ax.set_ylim(0, 1)
        return
    ymax = float(vals.max(skipna=True))
    if ymax <= 0:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, ymax * (1.0 + pad_ratio))


def _style_step_axes(ax) -> None:
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, prune=None))


def _yaxis_scientific_1e(ax) -> None:
    """Compact y labels (e.g. ``1.2e-4``) instead of long decimals."""
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False, useMathText=False)


def _yaxis_sci_for_energy_or_emissions(
    *,
    y_axis_label: str | None = None,
    codecarbon_column: str | None = None,
) -> bool:
    """Use 1e-style y ticks only for energy (kWh) and emissions — not duration, step index, etc."""
    if codecarbon_column in ("energy_consumed", "emissions"):
        return True
    if y_axis_label:
        s = y_axis_label.replace("\u2082", "2").casefold()
        return "kwh" in s or "co2e" in s or "co2" in s or "emission" in s
    return False


# Reserve the top of the figure for suptitle + caption so they do not overlap plot axes or each other.
_FIG_LAYOUT_RECT = (0.03, 0.05, 0.97, 0.78)


def _reserve_title_space(fig: Figure) -> None:
    eng = fig.get_layout_engine()
    if eng is None:
        return
    try:
        eng.set(rect=_FIG_LAYOUT_RECT)
    except (AttributeError, TypeError, ValueError):
        pass


def _fig_suptitle_and_caption(
    fig: Figure,
    title: str,
    caption: str,
    *,
    title_fontsize: int = 13,
) -> None:
    """Place main title and subtitle (caption) with vertical spacing from title line count."""
    _reserve_title_space(fig)
    title_lines = title.count("\n") + 1
    fig.suptitle(title, fontsize=title_fontsize, fontweight="bold", y=0.99, va="top")
    # Caption sits below the suptitle block; lower when the title has more lines.
    cap_y = {1: 0.888, 2: 0.818, 3: 0.752}.get(title_lines, 0.695)
    fig.text(0.5, cap_y, caption, ha="center", va="top", fontsize=8.5, style="italic", color="0.38")


def plot_steps_overview(
    df: pd.DataFrame,
    *,
    main_title: str,
    caption: str,
    out_path: Path,
    x_col: str = "step",
) -> None:
    """Per-step: duration, energy, emissions, then cumulative energy + cumulative emissions (CodeCarbon CSV units)."""
    if x_col not in df.columns:
        raise SystemExit(f"Missing x column {x_col!r}")

    required_any = ("duration", "energy_consumed", "emissions")
    if not any(c in df.columns for c in required_any):
        raise SystemExit(f"Expected at least one of {required_any}; got columns {list(df.columns)}")

    df, n_warm = _trim_codecarbon_warmup(df, x_col)
    caption = _append_warmup_to_caption(caption, n_warm, cumulative_note=True)
    x = pd.to_numeric(df[x_col], errors="coerce")
    panels: list[tuple[str, np.ndarray, str]] = []

    if "duration" in df.columns:
        panels.append(
            (
                "Time for one training step\n(wall-clock, CodeCarbon task window)",
                pd.to_numeric(df["duration"], errors="coerce").to_numpy(),
                "Seconds",
            )
        )
    if "energy_consumed" in df.columns:
        e = pd.to_numeric(df["energy_consumed"], errors="coerce")
        panels.append(
            (
                "Electrical energy per training step\n(instantaneous step estimate)",
                e.to_numpy(),
                "Kilowatt-hours (kWh)",
            )
        )
    if "emissions" in df.columns:
        panels.append(
            (
                "CO₂-equivalent per training step\n(grid intensity × energy)",
                pd.to_numeric(df["emissions"], errors="coerce").to_numpy(),
                "Kilograms CO₂e",
            )
        )
    if "energy_consumed" in df.columns:
        e = pd.to_numeric(df["energy_consumed"], errors="coerce")
        panels.append(
            (
                "Cumulative electrical energy\n(sum of per-step kWh so far)",
                e.cumsum().to_numpy(),
                "Kilowatt-hours (kWh), cumulative",
            )
        )
    if "emissions" in df.columns:
        em = pd.to_numeric(df["emissions"], errors="coerce")
        panels.append(
            (
                "Cumulative CO₂-equivalent\n(sum of per-step emissions so far)",
                em.cumsum().to_numpy(),
                "Kilograms CO₂e, cumulative",
            )
        )

    n = len(panels)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(11, 2.45 * n), sharex=True, layout="constrained")
    if n == 1:
        axes = [axes]

    _fig_suptitle_and_caption(fig, main_title, caption)

    for ax, (panel_title, y, y_label) in zip(axes, panels):
        ax.plot(x, y, linewidth=1.0, color="C0")
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(panel_title, fontsize=10, loc="left", pad=6)
        _style_step_axes(ax)
        _xlim_after_warmup(ax, x, n_warm)
        _ylim_from_zero(ax, y)
        if _yaxis_sci_for_energy_or_emissions(y_axis_label=y_label):
            _yaxis_scientific_1e(ax)

    axes[-1].set_xlabel("Training step index (first logged step = 1)", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# CodeCarbon step/substep rows attribute energy to CPU, GPU, and RAM (kWh per interval).
HARDWARE_ENERGY_SPECS: tuple[tuple[str, str, str], ...] = (
    ("cpu_energy", "CPU", "#1b9e77"),
    ("gpu_energy", "GPU", "#d95f02"),
    ("ram_energy", "RAM", "#7570b3"),
)


def plot_step_energy_by_hardware(
    df: pd.DataFrame,
    *,
    main_title: str,
    caption: str,
    out_path: Path,
    x_col: str = "step",
) -> None:
    """CPU, GPU, and RAM kWh on shared axes: top = per-step lines, bottom = cumulative lines."""
    present = [(col, label, color) for col, label, color in HARDWARE_ENERGY_SPECS if col in df.columns]
    if not present:
        print(f"Skip hardware energy plot: no columns {[c for c, _, _ in HARDWARE_ENERGY_SPECS]} in data")
        return
    if x_col not in df.columns:
        raise SystemExit(f"Missing x column {x_col!r}")

    df, n_warm = _trim_codecarbon_warmup(df, x_col)
    caption = _append_warmup_to_caption(caption, n_warm, cumulative_note=True)
    x = pd.to_numeric(df[x_col], errors="coerce")
    x_np = x.to_numpy()

    fig, (ax_step, ax_cum) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, layout="constrained")
    _fig_suptitle_and_caption(
        fig,
        main_title + "\n(single plot per row: CPU, GPU, RAM overlaid)",
        caption,
        title_fontsize=12,
    )

    ymax_step = 0.0
    ymax_cum = 0.0
    for col, label, color in present:
        y = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        yc = np.cumsum(y)
        ax_step.plot(x_np, y, color=color, linewidth=1.05, alpha=0.92, label=label)
        ax_cum.plot(x_np, yc, color=color, linewidth=1.1, alpha=0.92, label=label)
        ymax_step = max(ymax_step, float(np.nanmax(y)) if y.size else 0.0)
        ymax_cum = max(ymax_cum, float(np.nanmax(yc)) if yc.size else 0.0)

    ax_step.set_title("Electrical energy per training step (CodeCarbon: CPU vs GPU vs RAM)", fontsize=10, loc="left", pad=8)
    ax_step.set_ylabel("kWh / step", fontsize=9)
    ax_step.legend(loc="upper right", fontsize=9, title="Component", framealpha=0.92, ncol=min(3, len(present)))
    _style_step_axes(ax_step)
    _xlim_after_warmup(ax_step, x, n_warm)
    _ylim_from_zero(ax_step, np.array([ymax_step]) if ymax_step > 0 else np.array([0.0]))
    _yaxis_scientific_1e(ax_step)

    ax_cum.set_title("Cumulative electrical energy by component", fontsize=10, loc="left", pad=8)
    ax_cum.set_ylabel("kWh (cumulative)", fontsize=9)
    ax_cum.legend(loc="upper left", fontsize=9, title="Component", framealpha=0.92, ncol=min(3, len(present)))
    _style_step_axes(ax_cum)
    _xlim_after_warmup(ax_cum, x, n_warm)
    _ylim_from_zero(ax_cum, np.array([ymax_cum]) if ymax_cum > 0 else np.array([0.0]))
    _yaxis_scientific_1e(ax_cum)

    ax_cum.set_xlabel("Training step index", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


PHASE_LABELS_ORDER = (
    ("energy_forward", "Forward"),
    ("energy_backward", "Backward"),
    ("energy_optim", "Optimizer"),
)
PHASE_COLORS = {"energy_forward": "#2c7bb6", "energy_backward": "#fdae61", "energy_optim": "#7fbc41"}

# Wall time (seconds) from merged step + substep tables — same hues as energy phases for consistency.
DURATION_PHASE_LABELS_ORDER = (
    ("duration_forward_s", "Forward"),
    ("duration_backward_s", "Backward"),
    ("duration_optimizer_s", "Optimizer"),
)
DURATION_PHASE_COLORS = {
    "duration_forward_s": "#2c7bb6",
    "duration_backward_s": "#fdae61",
    "duration_optimizer_s": "#7fbc41",
}

DURATION_TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Whole step", "whole_step_duration_s"),
    ("Forward", "duration_forward_s"),
    ("Backward", "duration_backward_s"),
    ("Optimizer", "duration_optimizer_s"),
    ("Σ subphases", "duration_substeps_total_s"),
    ("Whole − Σ", "delta_step_minus_substeps_s"),
)

DURATION_PLOT_VALUE_COLUMNS: tuple[str, ...] = (
    "whole_step_duration_s",
    "duration_forward_s",
    "duration_backward_s",
    "duration_optimizer_s",
    "duration_substeps_total_s",
    "delta_step_minus_substeps_s",
)


def durations_seconds_to_ms_frame(work_s: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``work_s`` with duration columns scaled to milliseconds for plotting."""
    out = work_s.copy()
    for c in DURATION_PLOT_VALUE_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * 1000.0
    return out


def duration_mean_table_row(work_ms: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Header labels and one numeric row for means; ``work_ms`` is warmup-trimmed with values in ms."""
    headers: list[str] = []
    cells: list[str] = []
    for h, col in DURATION_TABLE_COLUMNS:
        if col not in work_ms.columns:
            continue
        headers.append(h)
        m = pd.to_numeric(work_ms[col], errors="coerce").mean()
        cells.append(f"{float(m):.3f}" if pd.notna(m) else "—")
    return headers, cells


def plot_per_step_total_and_cumulative(
    df: pd.DataFrame,
    *,
    main_title: str,
    caption: str,
    out_path: Path,
    energy_col: str,
    x_col: str = "step",
) -> None:
    """Two panels: mean energy per step, then cumulative."""
    if x_col not in df.columns or energy_col not in df.columns:
        raise SystemExit(f"Need columns {x_col!r} and {energy_col!r}")
    df, n_warm = _trim_codecarbon_warmup(df, x_col)
    caption = _append_warmup_to_caption(caption, n_warm, cumulative_note=True)
    x = pd.to_numeric(df[x_col], errors="coerce")
    e = pd.to_numeric(df[energy_col], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, layout="constrained")
    _fig_suptitle_and_caption(fig, main_title, caption)

    y1 = e.to_numpy()
    axes[0].plot(x, y1, linewidth=1.0, color="C0")
    axes[0].set_title(
        "Total electrical energy per step\n(all phases combined)",
        fontsize=10,
        loc="left",
        pad=6,
    )
    axes[0].set_ylabel("kWh (per step)", fontsize=9)
    _style_step_axes(axes[0])
    _xlim_after_warmup(axes[0], x, n_warm)
    _ylim_from_zero(axes[0], y1)
    _yaxis_scientific_1e(axes[0])

    y2 = e.cumsum().to_numpy()
    axes[1].plot(x, y2, linewidth=1.0, color="C2")
    axes[1].set_title("Cumulative electrical energy", fontsize=10, loc="left", pad=6)
    axes[1].set_ylabel("kWh (cumulative)", fontsize=9)
    _style_step_axes(axes[1])
    _xlim_after_warmup(axes[1], x, n_warm)
    _ylim_from_zero(axes[1], y2)
    _yaxis_scientific_1e(axes[1])

    axes[1].set_xlabel("Training step index", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_step_substep_durations(
    merged: pd.DataFrame,
    *,
    main_title: str,
    caption: str,
    out_path: Path,
    x_col: str = "step",
) -> None:
    """Whole-step wall time vs summed subphase times (lines), plus stacked bars; y-axis and table in ms."""
    if x_col not in merged.columns:
        raise SystemExit(f"Missing x column {x_col!r}")
    if "whole_step_duration_s" not in merged.columns:
        raise SystemExit("Merged duration frame needs whole_step_duration_s")
    ordered: list[tuple[str, str]] = []
    for col, human in DURATION_PHASE_LABELS_ORDER:
        if col in merged.columns:
            ordered.append((col, human))
    if not ordered:
        raise SystemExit("No duration_forward_s / duration_backward_s / duration_optimizer_s columns")

    work_s, n_skip = _trim_codecarbon_warmup(merged, x_col)
    cap = _append_warmup_to_caption(caption, n_skip, cumulative_note=False) + (
        " Y-axis and table: milliseconds (source CSVs use seconds)."
    )

    work = durations_seconds_to_ms_frame(work_s)
    step = pd.to_numeric(work[x_col], errors="coerce")
    xs = step
    whole = pd.to_numeric(work["whole_step_duration_s"], errors="coerce")

    fig = plt.figure(figsize=(11, 7.65), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[2.25, 2.25, 0.72], hspace=0.26)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax_tbl = fig.add_subplot(gs[2])
    _fig_suptitle_and_caption(fig, main_title, cap, title_fontsize=12)

    ax0.plot(step, whole, color="0.15", linewidth=1.15, label="Whole training step (CodeCarbon task)", alpha=0.95)
    if "duration_substeps_total_s" in work.columns:
        sub_tot = pd.to_numeric(work["duration_substeps_total_s"], errors="coerce")
        ax0.plot(step, sub_tot, color="C0", linewidth=1.05, linestyle="--", label="Sum of subphase durations", alpha=0.9)
    if "delta_step_minus_substeps_s" in work.columns:
        delta = pd.to_numeric(work["delta_step_minus_substeps_s"], errors="coerce")
        ax0.plot(step, delta, color="C3", linewidth=0.9, alpha=0.75, label="Whole step − sum(subphases)")
    ax0.set_ylabel("Milliseconds", fontsize=9)
    ax0.set_title("Wall time: whole step vs subphases", fontsize=10, loc="left", pad=6)
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.92)
    _style_step_axes(ax0)
    if n_skip and xs.notna().any():
        xa, xb = float(xs.min()), float(xs.max())
        span = max(xb - xa, 1.0)
        ax0.set_xlim(max(0.0, xa - 0.02 * span), xb + 0.02 * span)
    else:
        _xlim_from_zero(ax0, xs)
    ys_lines: list[np.ndarray] = [whole.to_numpy()]
    if "duration_substeps_total_s" in work.columns:
        ys_lines.append(pd.to_numeric(work["duration_substeps_total_s"], errors="coerce").to_numpy())
    if "delta_step_minus_substeps_s" in work.columns:
        ys_lines.append(pd.to_numeric(work["delta_step_minus_substeps_s"], errors="coerce").to_numpy())
    flat = np.concatenate(ys_lines)
    if np.isfinite(flat).any():
        lo = min(0.0, float(np.nanmin(flat)))
        hi = float(np.nanmax(flat))
        span = hi - lo
        pad = 0.06 * span if span > 0 else 0.06
        ax0.set_ylim(lo - pad if lo < 0 else 0.0, max(hi + pad, 1e-6))
    else:
        ax0.set_ylim(0, 1)

    step_arr = step.to_numpy()
    bottoms = np.zeros(len(step_arr))
    for col, human in ordered:
        vals = pd.to_numeric(work[col], errors="coerce").fillna(0.0).to_numpy()
        color = DURATION_PHASE_COLORS.get(col, None)
        ax1.bar(step_arr, vals, bottom=bottoms, label=human, width=0.85, alpha=0.9, color=color)
        bottoms = bottoms + vals
    ax1.set_xlabel("Training step index", fontsize=10)
    ax1.set_ylabel("Milliseconds (stacked)\nforward + backward + optimizer", fontsize=10)
    ax1.set_title("Time attributed to each subphase (CodeCarbon)", fontsize=10, loc="left", pad=6)
    ax1.legend(loc="upper right", fontsize=9, title="Phase", framealpha=0.92)
    _style_step_axes(ax1)
    if n_skip and xs.notna().any():
        xa, xb = float(xs.min()), float(xs.max())
        span = max(xb - xa, 1.0)
        ax1.set_xlim(max(0.0, xa - 0.02 * span), xb + 0.02 * span)
    else:
        _xlim_from_zero(ax1, xs)
    _ylim_from_zero(ax1, bottoms if bottoms.size else np.array([0.0]))

    ax_tbl.axis("off")
    th, tv = duration_mean_table_row(work)
    if th:
        tab = ax_tbl.table(
            cellText=[tv],
            colLabels=th,
            loc="center",
            cellLoc="center",
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.5)
        tab.scale(1.08, 1.45)
        for (ri, ci), cell in tab.get_celld().items():
            if ri == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#f0f0f0")
            else:
                cell.set_facecolor("#ffffff")
        warm = f"first {n_skip} step(s) excluded" if n_skip else "all steps"
        ax_tbl.set_title(f"Mean wall time (ms) — {warm}", fontsize=9.5, pad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def load_and_prepare_steps_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "task_name" not in df.columns:
        raise SystemExit(f"{path}: expected column 'task_name'")
    steps = df["task_name"].map(_parse_step_index)
    out = df.assign(step=steps).dropna(subset=["step"])
    out["step"] = out["step"].astype(int)
    return out.sort_values("step").reset_index(drop=True)


def load_and_aggregate_substeps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "task_name" not in df.columns:
        raise SystemExit(f"{path}: expected column 'task_name'")
    rows = []
    for _, r in df.iterrows():
        p = _parse_substep(r["task_name"])
        if p is None:
            continue
        phase, step = p
        rows.append({"step": step, "phase": phase, **r.to_dict()})
    if not rows:
        raise SystemExit(f"{path}: no parsable substep task names")
    long = pd.DataFrame(rows)
    # Wide table: one row per step, energy per phase
    if "energy_consumed" not in long.columns:
        raise SystemExit(f"{path}: missing energy_consumed")
    pivot_e = long.pivot_table(
        index="step",
        columns="phase",
        values="energy_consumed",
        aggfunc="sum",
    ).fillna(0.0)
    pivot_e = pivot_e.reindex(sorted(long["step"].unique())).sort_index()
    # Total step energy (sum of phases)
    total = pivot_e.sum(axis=1)
    out = pivot_e.reset_index()
    out["energy_total"] = total.values
    rename_map = {
        "forward_pass": "energy_forward",
        "backward_pass": "energy_backward",
        "optimisation_step": "energy_optim",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def plot_substeps_stacked_energy(
    df_wide: pd.DataFrame,
    *,
    main_title: str,
    caption: str,
    out_path: Path,
) -> None:
    """Stacked bars: mean kWh in forward vs backward vs optimizer within each step."""
    df_wide, n_warm = _trim_codecarbon_warmup(df_wide, "step")
    caption = _append_warmup_to_caption(caption, n_warm, cumulative_note=False)
    step = df_wide["step"].to_numpy()
    ordered: list[tuple[str, str]] = []
    for col, human in PHASE_LABELS_ORDER:
        if col in df_wide.columns:
            ordered.append((col, human))
    if not ordered:
        raise SystemExit("No per-phase energy columns to stack (expected forward/backward/optim)")

    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    _fig_suptitle_and_caption(fig, main_title, caption)

    bottoms = np.zeros(len(step))
    for col, human in ordered:
        vals = pd.to_numeric(df_wide[col], errors="coerce").fillna(0.0).to_numpy()
        color = PHASE_COLORS.get(col, None)
        ax.bar(step, vals, bottom=bottoms, label=human, width=0.85, alpha=0.9, color=color)
        bottoms = bottoms + vals

    ax.set_xlabel("Training step index", fontsize=10)
    ax.set_ylabel("Electrical energy (kWh)\nstacked: forward + backward + optimizer", fontsize=10)
    ax.set_title(
        "How much of each step’s electricity is attributed to each phase\n(same totals as the “per step” line above, split by CodeCarbon tasks)",
        fontsize=10,
        loc="left",
        pad=10,
    )
    ax.legend(loc="upper right", fontsize=9, title="Phase", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    xs = pd.Series(step)
    _xlim_after_warmup(ax, xs, n_warm)
    _ylim_from_zero(ax, bottoms if bottoms.size else np.array([0.0]))
    _yaxis_scientific_1e(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_substeps_phases_separate(
    df_wide: pd.DataFrame,
    *,
    main_title: str,
    caption: str,
    out_path: Path,
) -> None:
    """One figure: forward / backward / optimizer each in its own column (per-step + cumulative)."""
    df_wide, n_warm = _trim_codecarbon_warmup(df_wide, "step")
    caption = _append_warmup_to_caption(caption, n_warm, cumulative_note=True)
    step = df_wide["step"].to_numpy()
    xs = pd.Series(step)
    ordered: list[tuple[str, str]] = []
    for col, human in PHASE_LABELS_ORDER:
        if col in df_wide.columns:
            ordered.append((col, human))
    if not ordered:
        raise SystemExit("No per-phase energy columns (expected forward/backward/optim)")

    n_cols = len(ordered)
    fig, axes2d = plt.subplots(
        2,
        n_cols,
        figsize=(max(11.0, 3.7 * n_cols), 6.9),
        sharex=True,
        layout="constrained",
    )
    if n_cols == 1:
        axes_top = [axes2d[0]]
        axes_bot = [axes2d[1]]
    else:
        axes_top = list(axes2d[0, :])
        axes_bot = list(axes2d[1, :])

    _fig_suptitle_and_caption(
        fig,
        main_title + "\n(one column per substep; top = per step, bottom = cumulative)",
        caption,
        title_fontsize=12,
    )

    for j, (col, human) in enumerate(ordered):
        y = pd.to_numeric(df_wide[col], errors="coerce").fillna(0.0).to_numpy()
        yc = np.cumsum(y)
        color = PHASE_COLORS.get(col, f"C{j}")
        axt = axes_top[j]
        axb = axes_bot[j]
        axt.plot(step, y, color=color, linewidth=1.0, alpha=0.92)
        axt.set_title(f"{human}\nkWh per training step", fontsize=10)
        axt.set_ylabel("kWh / step", fontsize=9)
        _style_step_axes(axt)
        _xlim_after_warmup(axt, xs, n_warm)
        _ylim_from_zero(axt, y)
        _yaxis_scientific_1e(axt)

        axb.plot(step, yc, color=color, linewidth=1.05, alpha=0.92)
        axb.set_title(f"{human}\ncumulative kWh", fontsize=10)
        axb.set_ylabel("kWh (sum)", fontsize=9)
        _style_step_axes(axb)
        _xlim_after_warmup(axb, xs, n_warm)
        _ylim_from_zero(axb, yc)
        _yaxis_scientific_1e(axb)

    axes_bot[-1].set_xlabel("Training step index", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


METRIC_LABELS = {
    "energy_consumed": "Total electrical energy (kWh)\nentire training run, wall-clock window",
    "emissions": "Total CO₂-equivalent (kg)\nfrom grid intensity × energy",
    "duration": "Total wall time (seconds)\nCodeCarbon tracking span for full run",
}


def plot_full_summary_bars(
    df: pd.DataFrame,
    *,
    main_title: str,
    caption: str,
    out_path: Path,
    x_tick_labels: list[str] | None = None,
) -> None:
    """Whole-run CodeCarbon aggregates (often one bar per run or one mean bar)."""
    if df.empty:
        raise SystemExit("full summary CSV is empty")
    metrics = [c for c in ("energy_consumed", "emissions", "duration") if c in df.columns]
    if not metrics:
        raise SystemExit(f"No energy_consumed/emissions/duration in columns: {list(df.columns)} ")
    if x_tick_labels is not None:
        if len(x_tick_labels) != len(df):
            raise SystemExit("x_tick_labels length must match number of rows in summary CSV")
        x_labels = x_tick_labels
    else:
        x_labels = []
        for i, r in df.iterrows():
            rid = r.get("run_id", f"row_{i}")
            x_labels.append(str(rid)[:8])

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(7.0, 1.2 * len(df)), 2.95 * len(metrics)), layout="constrained")
    if len(metrics) == 1:
        axes = [axes]
    _fig_suptitle_and_caption(fig, main_title, caption)

    x = np.arange(len(df), dtype=float)
    width = min(0.5, 0.8 / max(len(df), 1))
    for ax, m in zip(axes, metrics):
        vals = pd.to_numeric(df[m], errors="coerce").to_numpy()
        ax.bar(x, vals, width=width, color="C0", alpha=0.88, align="center")
        ax.set_ylabel(METRIC_LABELS.get(m, m), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.set_xlim(left=-0.6, right=max(len(df) - 1, 0) + 0.6)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)
        _ylim_from_zero(ax, vals, pad_ratio=0.08)
        if _yaxis_sci_for_energy_or_emissions(codecarbon_column=m):
            _yaxis_scientific_1e(ax)

    axes[-1].set_xlabel("Run or aggregate (tick labels)", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _infer_kind(path: Path) -> str:
    name = path.name
    if name.endswith("-steps.csv"):
        return "steps"
    if name.endswith("-substeps.csv"):
        return "substeps"
    # With ``run_<n>_`` prefix: ``..._cc_full_...``; E2E baseline uses ``..._cc_e2e_full_...``
    if "_cc_full_" in name or name.startswith("cc_full_"):
        return "full"
    if "_cc_e2e_full_" in name or name.startswith("cc_e2e_full_"):
        return "full"
    if "_cc_step_" in name or name.startswith("cc_step_"):
        return "steps"  # user may pass base file; prefer sibling -steps
    if "_cc_substep_" in name or name.startswith("cc_substep_"):
        return "substeps"
    return "unknown"


def resolve_task_csv(path: Path) -> Path:
    """If ``path`` is base ``cc_step`` / ``cc_substep`` CSV, return sibling task CSV."""
    kind = _infer_kind(path)
    if path.suffix != ".csv":
        return path
    stem, ext = path.stem, path.suffix
    if kind == "steps" and not stem.endswith("-steps"):
        candidate = path.with_name(stem + "-steps" + ext)
        if candidate.is_file():
            return candidate
    if kind == "substeps" and not stem.endswith("-substeps"):
        candidate = path.with_name(stem + "-substeps" + ext)
        if candidate.is_file():
            return candidate
    return path


def plot_one_file(path: Path, plots_root: Path, output: Path | None) -> None:
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    task_path = resolve_task_csv(path)
    kind = _infer_kind(task_path)

    out_dir = _mirror_plots_subdir(task_path, plots_root)
    base_stem = task_path.stem

    table_path = out_dir / f"{base_stem}_codecarbon_summary_table.csv"
    summary_rows: list[dict[str, str | int | float]] = []

    if kind == "steps":
        df = load_and_prepare_steps_csv(task_path)
        outp = output if output else out_dir / f"{base_stem}_codecarbon_metrics.png"
        plot_steps_overview(
            df,
            main_title=f"CodeCarbon — metrics per training step\n{task_path.name}",
            caption="Single run: each point is one optimizer step (CodeCarbon task per step).",
            out_path=outp,
            x_col="step",
        )
        plot_step_energy_by_hardware(
            df,
            main_title=f"CodeCarbon — energy by hardware component\n{task_path.name}",
            caption="Single run. kWh attributed to CPU, GPU, and RAM per step (CodeCarbon estimates).",
            out_path=out_dir / f"{base_stem}_codecarbon_hardware_energy.png",
            x_col="step",
        )
        summary_rows.append(
            build_steps_summary_row(
                df,
                section="steps_rollup",
                source_file=task_path.name,
                n_runs_averaged=1,
                notes="Single run.",
            )
        )
        pair_sub = resolve_paired_substeps_csv_from_steps_csv(task_path)
        if pair_sub is not None:
            write_merged_duration_if_paired(
                out_dir,
                df,
                [pair_sub],
                stem_prefix=f"{base_stem}_",
                duration_plot_title=f"CodeCarbon — step vs subphase duration\n{task_path.name}",
                duration_plot_caption="Single run: whole step task vs CodeCarbon subphases (figure in ms).",
            )

    elif kind == "substeps":
        wide = load_and_aggregate_substeps(task_path)
        outp = output if output else out_dir / f"{base_stem}_codecarbon_subphase_energy.png"
        plot_substeps_stacked_energy(
            wide,
            main_title=f"CodeCarbon — electricity by phase\n{task_path.name}",
            caption="Single run: bar height = forward + backward + optimizer kWh for that step (CodeCarbon substeps).",
            out_path=outp,
        )
        plot_substeps_phases_separate(
            wide,
            main_title=f"CodeCarbon — electricity by training substep\n{task_path.name}",
            caption="Single run: each phase (forward / backward / optimizer) in its own column.",
            out_path=outp.with_name(outp.stem + "_isolated.png"),
        )
        line_out = outp.with_name(outp.stem + "_total_energy.png")
        plot_per_step_total_and_cumulative(
            wide.rename(columns={"energy_total": "energy_consumed"}),
            main_title=f"CodeCarbon — total energy per step\n{task_path.name}",
            caption="Per-step total matches stacked bars above; cumulative sums those kWh in order.",
            energy_col="energy_consumed",
            out_path=line_out,
            x_col="step",
        )
        summary_rows.append(
            build_substeps_summary_row(
                wide,
                section="substeps_rollup",
                source_file=task_path.name,
                n_runs_averaged=1,
                notes="Single run.",
            )
        )
        pair_step = resolve_paired_steps_csv_from_substeps_csv(task_path)
        if pair_step is not None:
            step_df_single = load_and_prepare_steps_csv(pair_step)
            write_merged_duration_if_paired(
                out_dir,
                step_df_single,
                [task_path],
                stem_prefix=f"{base_stem}_",
                duration_plot_title=(
                    f"CodeCarbon — step vs subphase duration\n{pair_step.name} + {task_path.name}"
                ),
                duration_plot_caption="Single run: paired step and substep logs (figure in ms).",
            )

    elif kind == "full":
        df = pd.read_csv(task_path)
        outp = output if output else out_dir / f"{base_stem}_codecarbon_summary.png"
        m = RUN_PREFIX_RE.match(task_path.name)
        run_idx = int(m.group(1)) if m else None
        summary_rows.extend(build_full_run_rows_from_dataframe(df, task_path.name, run_index=run_idx))
        if len(df) < 2:
            print(
                f"Skip plot for {task_path.name}: only one row — use cumulative curves on the "
                "per-step plot for whole-run energy/CO₂e (or merge multiple runs into one CSV to compare)."
            )
        else:
            plot_full_summary_bars(
                df,
                main_title=f"CodeCarbon — whole-run totals\n{task_path.name}",
                caption="One bar group per row (several runs in this file).",
                out_path=outp,
            )

    else:
        raise SystemExit(
            f"Unrecognised CodeCarbon CSV type: {path.name!r} "
            "(expected *cc_full*, *cc_e2e_full*, *cc_step* / *-steps*, *cc_substep* / *-substeps*)"
        )

    if summary_rows:
        write_codecarbon_summary_table(table_path, summary_rows)


def average_steps_group(paths: list[Path]) -> pd.DataFrame:
    dfs = [load_and_prepare_steps_csv(p.resolve()) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)
    out = combined.groupby("step", as_index=False).mean(numeric_only=True)
    return out.sort_values("step").reset_index(drop=True)


def average_substeps_group(paths: list[Path]) -> pd.DataFrame:
    dfs = [load_and_aggregate_substeps(p.resolve()) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)
    value_cols = [c for c in combined.columns if c != "step"]
    out = combined.groupby("step", as_index=False)[value_cols].mean(numeric_only=True)
    return out.sort_values("step").reset_index(drop=True)


def load_and_aggregate_substep_durations(path: Path) -> pd.DataFrame:
    """One row per training step: duration (s) for forward, backward, optimizer (CodeCarbon tasks)."""
    df = pd.read_csv(path)
    if "task_name" not in df.columns:
        raise SystemExit(f"{path}: expected column 'task_name'")
    rows: list[dict] = []
    for _, r in df.iterrows():
        p = _parse_substep(r["task_name"])
        if p is None:
            continue
        phase, step = p
        rows.append({"step": step, "phase": phase, **r.to_dict()})
    if not rows:
        raise SystemExit(f"{path}: no parsable substep task names")
    long = pd.DataFrame(rows)
    if "duration" not in long.columns:
        raise SystemExit(f"{path}: missing duration")
    pivot = long.pivot_table(index="step", columns="phase", values="duration", aggfunc="sum").fillna(0.0)
    pivot = pivot.reindex(sorted(long["step"].unique())).sort_index()
    total = pivot.sum(axis=1)
    out = pivot.reset_index()
    out["duration_substeps_total_s"] = total.values
    rename_map = {
        "forward_pass": "duration_forward_s",
        "backward_pass": "duration_backward_s",
        "optimisation_step": "duration_optimizer_s",
    }
    return out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})


def average_substeps_duration_group(paths: list[Path]) -> pd.DataFrame:
    dfs = [load_and_aggregate_substep_durations(p.resolve()) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)
    value_cols = [c for c in combined.columns if c != "step"]
    out = combined.groupby("step", as_index=False)[value_cols].mean(numeric_only=True)
    return out.sort_values("step").reset_index(drop=True)


def resolve_paired_substeps_csv_from_steps_csv(step_csv: Path) -> Path | None:
    """``run_K_cc_step_...-steps.csv`` → sibling ``run_K_cc_substep_...-substeps.csv`` if present."""
    name = step_csv.name
    if "-steps.csv" not in name or "cc_step_" not in name:
        return None
    sub_name = name.replace("cc_step_", "cc_substep_").replace("-steps.csv", "-substeps.csv")
    cand = step_csv.with_name(sub_name)
    return cand if cand.is_file() else None


def resolve_paired_steps_csv_from_substeps_csv(sub_csv: Path) -> Path | None:
    """``run_K_cc_substep_...-substeps.csv`` → sibling ``run_K_cc_step_...-steps.csv`` if present."""
    name = sub_csv.name
    if "-substeps.csv" not in name or "cc_substep_" not in name:
        return None
    step_name = name.replace("cc_substep_", "cc_step_").replace("-substeps.csv", "-steps.csv")
    cand = sub_csv.with_name(step_name)
    if not cand.is_file():
        return None
    return resolve_task_csv(cand)


def merge_duration_by_step(step_df: pd.DataFrame, substep_dur_df: pd.DataFrame) -> pd.DataFrame:
    """Align whole-step duration with per-phase substep durations (seconds)."""
    s = step_df[["step", "duration"]].copy()
    s = s.rename(columns={"duration": "whole_step_duration_s"})
    m = s.merge(substep_dur_df, on="step", how="outer", sort=True)
    w = pd.to_numeric(m["whole_step_duration_s"], errors="coerce")
    t = pd.to_numeric(m["duration_substeps_total_s"], errors="coerce") if "duration_substeps_total_s" in m else None
    if t is not None:
        m["delta_step_minus_substeps_s"] = w - t
    return m.reset_index(drop=True)


def build_duration_summary(merged: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in merged.columns if c != "step"]
    rows: list[dict[str, str | float]] = []
    for stat_name, fn in (
        ("mean", np.nanmean),
        ("median", np.nanmedian),
        ("std", np.nanstd),
        ("min", np.nanmin),
        ("max", np.nanmax),
    ):
        row: dict[str, str | float] = {"statistic": stat_name}
        for c in cols:
            arr = pd.to_numeric(merged[c], errors="coerce").to_numpy()
            row[c] = float(fn(arr)) if np.isfinite(arr).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_duration_table_files(out_dir: Path, merged: pd.DataFrame, *, stem_prefix: str) -> None:
    """Write per-step and summary duration CSVs; ``stem_prefix`` ends with ``_`` or is empty."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_step = out_dir / f"{stem_prefix}codecarbon_duration_by_step.csv"
    summary_path = out_dir / f"{stem_prefix}codecarbon_duration_summary.csv"
    summ = build_duration_summary(merged)
    merged.to_csv(by_step, index=False)
    summ.to_csv(summary_path, index=False)
    print("\n--- CodeCarbon duration tables (seconds) ---")
    print(summ.to_string(index=False))
    print(f"\nWrote {by_step}\nWrote {summary_path}\n")


def write_merged_duration_if_paired(
    out_dir: Path,
    step_df: pd.DataFrame,
    substep_paths: list[Path],
    *,
    stem_prefix: str,
    duration_plot_title: str | None = None,
    duration_plot_caption: str | None = None,
) -> None:
    if not substep_paths:
        return
    try:
        if len(substep_paths) > 1:
            sub_dur = average_substeps_duration_group(substep_paths)
        else:
            sub_dur = load_and_aggregate_substep_durations(substep_paths[0].resolve())
    except SystemExit as e:
        print(f"Skip duration tables: {e}")
        return
    merged = merge_duration_by_step(step_df, sub_dur)
    write_duration_table_files(out_dir, merged, stem_prefix=stem_prefix)
    plot_path = out_dir / f"{stem_prefix}codecarbon_duration.png"
    pfx = stem_prefix.rstrip("_") or "duration"
    cap = (
        "Whole-step CodeCarbon task vs subphase durations (plotted in ms; CSVs in s)."
        if duration_plot_caption is None
        else duration_plot_caption
    )
    plot_step_substep_durations(
        merged,
        main_title=duration_plot_title
        or f"CodeCarbon — step vs subphase duration (wall time, ms)\n{pfx}",
        caption=cap,
        out_path=plot_path,
    )


SUMMARY_COLUMNS = [
    "section",
    "source_file",
    "n_runs_averaged",
    "training_steps_logged",
    "mean_step_duration_s",
    "sum_per_step_wall_time_s",
    "total_energy_kwh",
    "total_emissions_kg_co2e",
    "full_run_duration_s",
    "energy_forward_kwh",
    "energy_backward_kwh",
    "energy_optimizer_kwh",
    "notes",
]


def build_steps_summary_row(
    df: pd.DataFrame,
    *,
    section: str,
    source_file: str,
    n_runs_averaged: int,
    notes: str = "",
) -> dict[str, str | int | float]:
    dur = pd.to_numeric(df["duration"], errors="coerce") if "duration" in df.columns else pd.Series(dtype=float)
    e = pd.to_numeric(df["energy_consumed"], errors="coerce") if "energy_consumed" in df.columns else pd.Series(dtype=float)
    em = pd.to_numeric(df["emissions"], errors="coerce") if "emissions" in df.columns else pd.Series(dtype=float)
    step_max = int(df["step"].max()) if "step" in df.columns and len(df) else np.nan
    return {
        "section": section,
        "source_file": source_file,
        "n_runs_averaged": n_runs_averaged,
        "training_steps_logged": step_max,
        "mean_step_duration_s": float(dur.mean()) if dur.notna().any() else np.nan,
        "sum_per_step_wall_time_s": float(dur.sum()) if dur.notna().any() else np.nan,
        "total_energy_kwh": float(e.sum()) if e.notna().any() else np.nan,
        "total_emissions_kg_co2e": float(em.sum()) if em.notna().any() else np.nan,
        "full_run_duration_s": np.nan,
        "energy_forward_kwh": np.nan,
        "energy_backward_kwh": np.nan,
        "energy_optimizer_kwh": np.nan,
        "notes": notes
        or (
            "Sums are over the per-step series (mean across runs when n_runs_averaged>1); "
            "not identical to the mean of each run’s total from cc_full."
        ),
    }


def build_substeps_summary_row(
    wide: pd.DataFrame,
    *,
    section: str,
    source_file: str,
    n_runs_averaged: int,
    notes: str = "",
) -> dict[str, str | int | float]:
    def col_sum(name: str) -> float:
        if name not in wide.columns:
            return np.nan
        return float(pd.to_numeric(wide[name], errors="coerce").fillna(0.0).sum())

    et = col_sum("energy_total")
    step_max = int(wide["step"].max()) if "step" in wide.columns and len(wide) else np.nan
    return {
        "section": section,
        "source_file": source_file,
        "n_runs_averaged": n_runs_averaged,
        "training_steps_logged": step_max,
        "mean_step_duration_s": np.nan,
        "sum_per_step_wall_time_s": np.nan,
        "total_energy_kwh": et,
        "total_emissions_kg_co2e": np.nan,
        "full_run_duration_s": np.nan,
        "energy_forward_kwh": col_sum("energy_forward"),
        "energy_backward_kwh": col_sum("energy_backward"),
        "energy_optimizer_kwh": col_sum("energy_optim"),
        "notes": notes
        or "Substep tasks only: kWh by phase summed over steps (mean curve when n_runs_averaged>1).",
    }


def build_full_run_rows_from_dataframe(
    df: pd.DataFrame,
    source_file: str,
    *,
    run_index: int | None = None,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for i, r in df.iterrows():
        rows.append(
            {
                "section": "full_run_file",
                "source_file": source_file,
                "n_runs_averaged": np.nan,
                "training_steps_logged": np.nan,
                "mean_step_duration_s": np.nan,
                "sum_per_step_wall_time_s": np.nan,
                "total_energy_kwh": float(pd.to_numeric(r.get("energy_consumed"), errors="coerce")),
                "total_emissions_kg_co2e": float(pd.to_numeric(r.get("emissions"), errors="coerce")),
                "full_run_duration_s": float(pd.to_numeric(r.get("duration"), errors="coerce")),
                "energy_forward_kwh": np.nan,
                "energy_backward_kwh": np.nan,
                "energy_optimizer_kwh": np.nan,
                "notes": f"CodeCarbon full-run row csv_index={i}, run_index={run_index}",
            }
        )
    return rows


def collect_full_run_table_rows(directory: Path) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    full_paths = sorted(
        {p.resolve() for p in directory.glob("*_cc_full_*.csv")}
        | {p.resolve() for p in directory.glob("*_cc_e2e_full_*.csv")},
        key=lambda p: p.name,
    )
    for p in full_paths:
        df = pd.read_csv(p)
        m = RUN_PREFIX_RE.match(p.name)
        run_idx = int(m.group(1)) if m else None
        rows.extend(build_full_run_rows_from_dataframe(df, p.name, run_index=run_idx))
    if len(rows) >= 2:
        n_files = len(rows)
        eng = [
            float(r["total_energy_kwh"])  # type: ignore[arg-type]
            for r in rows
            if pd.notna(r["total_energy_kwh"])
        ]
        emi = [
            float(r["total_emissions_kg_co2e"])  # type: ignore[arg-type]
            for r in rows
            if pd.notna(r["total_emissions_kg_co2e"])
        ]
        dur = [
            float(r["full_run_duration_s"])  # type: ignore[arg-type]
            for r in rows
            if pd.notna(r["full_run_duration_s"])
        ]
        rows.append(
            {
                "section": "full_run_mean_of_files",
                "source_file": "(arithmetic mean of full_run_file rows)",
                "n_runs_averaged": n_files,
                "training_steps_logged": np.nan,
                "mean_step_duration_s": np.nan,
                "sum_per_step_wall_time_s": np.nan,
                "total_energy_kwh": float(np.mean(eng)) if eng else np.nan,
                "total_emissions_kg_co2e": float(np.mean(emi)) if emi else np.nan,
                "full_run_duration_s": float(np.mean(dur)) if dur else np.nan,
                "energy_forward_kwh": np.nan,
                "energy_backward_kwh": np.nan,
                "energy_optimizer_kwh": np.nan,
                "notes": f"Mean of {n_files} cc_full rows listed above.",
            }
        )
    return rows


def write_codecarbon_summary_table(out_path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    for c in SUMMARY_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[SUMMARY_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    disp = df.copy()
    for c in disp.columns:
        if pd.api.types.is_numeric_dtype(disp[c]):
            disp[c] = disp[c].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
    print("\n--- CodeCarbon summary table ---")
    print(disp.to_string(index=False))
    print(f"\nWrote {out_path}\n")


def plot_directory_averaged(directory: Path, plots_root: Path) -> None:
    """Group ``run_*`` CSVs by shared suffix, mean across runs, write under ``results/plots/...``."""
    directory = directory.resolve()
    files = _discover_csvs(directory)
    if not files:
        raise SystemExit(f"No CodeCarbon CSVs found under {directory}")

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        groups[_template_key(p)].append(p)

    out_dir = _mirror_plots_subdir(files[0], plots_root)
    summary_rows: list[dict[str, str | int | float]] = collect_full_run_table_rows(directory)

    for template_key, paths in sorted(groups.items(), key=lambda kv: kv[0]):
        paths = sorted(paths, key=_run_sort_key)
        kind = _infer_kind(Path(template_key))
        base_stem = Path(template_key).stem
        n_runs = len(paths)

        if kind == "steps":
            df = average_steps_group(paths)
            plot_steps_overview(
                df,
                main_title=f"CodeCarbon — metrics per training step (mean of {n_runs} runs)\n{base_stem}",
                caption="",
                out_path=out_dir / f"{base_stem}_mean_codecarbon_metrics.png",
                x_col="step",
            )
            plot_step_energy_by_hardware(
                df,
                main_title=f"CodeCarbon — energy by hardware (mean of {n_runs} runs)\n{base_stem}",
                caption="Stacked kWh = CPU + GPU + RAM per step.",
                out_path=out_dir / f"{base_stem}_mean_codecarbon_hardware_energy.png",
                x_col="step",
            )
            summary_rows.append(
                build_steps_summary_row(
                    df,
                    section="steps_mean_rollup",
                    source_file=f"{base_stem} (mean of {n_runs} runs)",
                    n_runs_averaged=n_runs,
                )
            )
        elif kind == "substeps":
            wide = average_substeps_group(paths)
            stack_out = out_dir / f"{base_stem}_mean_codecarbon_subphase_energy.png"
            plot_substeps_stacked_energy(
                wide,
                main_title=f"CodeCarbon — electricity by phase (mean of {n_runs} runs)\n{base_stem}",
                caption="Stacked layers = forward / backward / optimizer.",
                out_path=stack_out,
            )
            plot_substeps_phases_separate(
                wide,
                main_title=f"CodeCarbon — electricity by training substep (mean of {n_runs} runs)\n{base_stem}",
                caption="One column per phase; easier to compare shape without stacking.",
                out_path=stack_out.with_name(stack_out.stem + "_isolated.png"),
            )
            line_out = out_dir / f"{base_stem}_mean_codecarbon_total_energy.png"
            plot_per_step_total_and_cumulative(
                wide.rename(columns={"energy_total": "energy_consumed"}),
                main_title=f"CodeCarbon — total energy per step (mean of {n_runs} runs)\n{base_stem}",
                caption="",
                energy_col="energy_consumed",
                out_path=line_out,
                x_col="step",
            )
            summary_rows.append(
                build_substeps_summary_row(
                    wide,
                    section="substeps_mean_rollup",
                    source_file=f"{base_stem} (mean of {n_runs} runs)",
                    n_runs_averaged=n_runs,
                )
            )
        else:
            raise SystemExit(f"Unrecognised CodeCarbon group template: {template_key!r}")

    for template_key in sorted(groups.keys()):
        if _infer_kind(Path(template_key)) != "steps":
            continue
        sub_tmpl = _paired_substep_template_key(template_key)
        if not sub_tmpl or sub_tmpl not in groups:
            continue
        step_paths = sorted(groups[template_key], key=_run_sort_key)
        sub_paths = sorted(groups[sub_tmpl], key=_run_sort_key)
        df_steps_mean = average_steps_group(step_paths)
        stem_prefix = f"{Path(template_key).stem}_"
        n_pair = len(step_paths)
        write_merged_duration_if_paired(
            out_dir,
            df_steps_mean,
            sub_paths,
            stem_prefix=stem_prefix,
            duration_plot_title=(
                f"CodeCarbon — step vs subphase duration (mean of {n_pair} runs)\n"
                f"{Path(template_key).stem}"
            ),
            duration_plot_caption="",
        )

    write_codecarbon_summary_table(out_dir / "codecarbon_summary_table.csv", summary_rows)


def _discover_csvs(directory: Path) -> list[Path]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")
    paths: list[Path] = []
    for pattern in (
        "*_cc_step_*-steps.csv",
        "*_cc_substep_*-substeps.csv",
    ):
        paths.extend(directory.glob(pattern))
    paths = sorted({p.resolve() for p in paths}, key=lambda p: p.name)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CodeCarbon trainer CSV logs.")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a CodeCarbon .csv or a directory containing run_*_cc_*.csv files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (single file only; ignored when input_path is a directory)",
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=DEFAULT_PLOTS_ROOT,
        help=f"Base directory for default outputs (default: {DEFAULT_PLOTS_ROOT})",
    )
    args = parser.parse_args()

    plots_root = args.plots_root
    if not plots_root.is_absolute():
        plots_root = Path.cwd() / plots_root

    inp = args.input_path.resolve()
    if inp.is_dir():
        plot_directory_averaged(inp, plots_root)
        return

    plot_one_file(inp, plots_root, output=args.output)


if __name__ == "__main__":
    main()

