#!/usr/bin/env python3
"""Compare trainer-stats artifacts across synthetic_whisper **data-type** result trees.

Uses **run index 1 only** (``run_1`` / ``memory_run_1`` filenames). For each backend
root and artifact type, issues ``warnings.warn`` when an expected file is missing.
If ``run_2``+ files exist under a cell, emits one informational warning that they
were skipped.

Example (after sweeps under separate roots):

  python3 scripts/compare_data_type_results.py \\
    --scan-parent results \\
    --batch 64 --worker 0

  python3 scripts/compare_data_type_results.py \\
    --root results/data_chunk --label chunks \\
    --root results/data_shard --label shard \\
    --root results/data_memmap --label memmap \\
    --root results/data_memory --label memory \\
    --batch 64 --worker 0 \\
    --out-dir results/plots/compare_data_types/batch_64_worker_0

Outputs (when corresponding inputs exist):

- ``compare_resource_util_run_1.png`` — per-step metrics, one line per backend
- ``compare_phase_timing_run_1.png`` — grouped bars by backend × phase (ms)
- ``compare_phase_timing_total_run_1.png`` — sum of available phase means per backend
- ``compare_train_duration_run_1.png`` — wall time from resource_util duration txt
- ``compare_codecarbon_coarse_run_1.png`` — duration / energy / emissions (full coarse)
- ``compare_codecarbon_e2e_energy_run_1.png`` — e2e ``energy_consumed`` (+ emissions)

By default, all figures use **ratios vs the ``memory`` backend** (that bar/line = 1).
Use ``--normalize-to ''`` for raw units. ``--normalize-to chunks`` selects another baseline.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from plot_phase_timing_bars import PHASE_SPEC, _mean_ms_from_row

RUN_INDEX = 1
RUN_TOKEN_NONMEM = f"run_{RUN_INDEX}_"
RUN_TOKEN_MEM = f"memory_run_{RUN_INDEX}_"

SCAN_LABEL_ALIASES = {
    "data_chunk": "chunks",
    "data_chunks": "chunks",
    "data_shard": "shard",
    "data_memmap": "memmap",
    "data_single_file": "single_file",
    "data_memory": "memory",
}


def _set_xlim_step_from_zero(ax, step: pd.Series, *, right_pad_ratio: float = 0.02) -> None:
    xs = pd.to_numeric(step, errors="coerce")
    if not xs.notna().any():
        ax.set_xlim(0, 1)
        return
    xmax = float(xs.max(skipna=True))
    if xmax <= 0:
        ax.set_xlim(0, 1)
    else:
        ax.set_xlim(0, xmax * (1.0 + right_pad_ratio))


def _ratio_vs_baseline(
    labels: list[str],
    values: list[float],
    baseline_label: str,
    *,
    context: str,
) -> tuple[list[float], bool]:
    """Return ``values / baseline`` when baseline label exists and is valid; else raw values."""
    if baseline_label not in labels:
        warnings.warn(
            f"{context}: baseline {baseline_label!r} missing; plotting raw values",
            UserWarning,
            stacklevel=2,
        )
        return list(values), False
    i = labels.index(baseline_label)
    base = float(values[i])
    if not np.isfinite(base) or base == 0:
        warnings.warn(
            f"{context}: baseline {baseline_label!r} value is not a positive finite number; "
            "plotting raw values",
            UserWarning,
            stacklevel=2,
        )
        return list(values), False
    return [float(v) / base for v in values], True


def _set_ylim_bottom_zero_with_headroom(ax, y: pd.Series, *, pad_ratio: float = 0.08) -> None:
    vals = pd.to_numeric(y, errors="coerce")
    if not vals.notna().any():
        ax.set_ylim(0, 1)
        return
    ymax = float(vals.max(skipna=True))
    if ymax <= 0:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, ymax * (1.0 + pad_ratio))


def _metric_columns(df: pd.DataFrame) -> list[str]:
    if "step" not in df.columns:
        return []
    return [c for c in df.columns if c != "step"]


def _uses_memory_prefix(label: str) -> bool:
    return label.strip().lower() == "memory"


def _warn_missing(backend_label: str, artifact: str, path: Path) -> None:
    warnings.warn(
        f"[{backend_label}] missing {artifact}: expected {path}",
        UserWarning,
        stacklevel=2,
    )


def _warn_extra_runs(cell_dir: Path, backend_label: str, patterns: list[str]) -> None:
    seen = False
    for pat in patterns:
        for p in cell_dir.glob(pat):
            m = re.search(r"_run_(\d+)\.", p.name)
            if m and int(m.group(1)) != RUN_INDEX:
                seen = True
                break
        if seen:
            break
    if seen:
        warnings.warn(
            f"[{backend_label}] ignoring run_2+ files under {cell_dir} (v1 uses run_{RUN_INDEX} only)",
            UserWarning,
            stacklevel=2,
        )


def _cell_dir(root: Path, batch: int, worker: int) -> Path:
    return root / f"batch_{batch}_worker_{worker}"


def _resolve_resource_util_csv(cell_dir: Path, backend_label: str) -> Path | None:
    mem = _uses_memory_prefix(backend_label)
    primary = cell_dir / (
        f"memory_resource_util_run_{RUN_INDEX}.csv"
        if mem
        else f"resource_util_run_{RUN_INDEX}.csv"
    )
    alt = cell_dir / (
        f"resource_util_run_{RUN_INDEX}.csv"
        if mem
        else f"memory_resource_util_run_{RUN_INDEX}.csv"
    )
    if primary.is_file():
        if alt.is_file() and alt != primary:
            warnings.warn(
                f"[{backend_label}] both {primary.name} and {alt.name} exist; using {primary.name}",
                UserWarning,
                stacklevel=2,
            )
        return primary
    if alt.is_file():
        warnings.warn(
            f"[{backend_label}] resource_util: expected {primary.name}, using {alt.name}",
            UserWarning,
            stacklevel=2,
        )
        return alt
    _warn_missing(backend_label, "resource_util", primary)
    return None


def _resolve_train_duration_txt(cell_dir: Path, backend_label: str) -> Path | None:
    mem = _uses_memory_prefix(backend_label)
    candidates: list[Path] = []
    if mem:
        candidates = [
            cell_dir / f"memory_resource_util_train_duration_run_{RUN_INDEX}.txt",
            cell_dir / "memory_resource_util_train_duration.txt",
        ]
    else:
        candidates = [
            cell_dir / f"resource_util_train_duration_run_{RUN_INDEX}.txt",
            cell_dir / "resource_util_train_duration.txt",
        ]
    for i, p in enumerate(candidates):
        if p.is_file():
            if i > 0:
                warnings.warn(
                    f"[{backend_label}] train_duration: preferred {candidates[0].name} missing, using {p.name}",
                    UserWarning,
                    stacklevel=2,
                )
            return p
    _warn_missing(backend_label, "train_duration", candidates[0])
    return None


def _parse_duration_ms(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8").strip()
    # "duration_ms 123.45"
    parts = text.replace("\n", " ").split()
    for i, tok in enumerate(parts):
        if tok == "duration_ms" and i + 1 < len(parts):
            try:
                return float(parts[i + 1])
            except ValueError:
                return None
    return None


def _glob_single(
    cell_dir: Path, backend_label: str, glob_pat: str, artifact: str
) -> Path | None:
    matches = sorted(cell_dir.glob(glob_pat))
    files = [p for p in matches if p.is_file()]
    if not files:
        _warn_missing(backend_label, artifact, cell_dir / glob_pat)
        return None
    if len(files) > 1:
        warnings.warn(
            f"[{backend_label}] {artifact}: multiple matches for {glob_pat!r}, using {files[0].name}",
            UserWarning,
            stacklevel=2,
        )
    return files[0]


def _collect_phase_timing_row(
    cell_dir: Path, backend_label: str
) -> dict[str, float] | None:
    """Return phase -> ms for run 1 using either unified or per-measure summary files."""
    pfx = RUN_TOKEN_MEM if _uses_memory_prefix(backend_label) else RUN_TOKEN_NONMEM

    unified = list(cell_dir.glob(f"{pfx}phase_timing_rank_*_summary.csv"))
    unified = [p for p in unified if p.is_file()]
    if len(unified) == 1:
        df = pd.read_csv(unified[0])
        if len(df) != 1:
            warnings.warn(
                f"[{backend_label}] phase_timing unified summary {unified[0]}: expected 1 row, got {len(df)}",
                UserWarning,
                stacklevel=2,
            )
        row = df.iloc[0]
        return {
            phase: _mean_ms_from_row(row, phase)
            for phase, _label, _c in PHASE_SPEC
        }

    out: dict[str, float] = {}
    for phase, _label, _color in PHASE_SPEC:
        pat = f"{pfx}phase_timing_measure_{phase}_rank_*_summary.csv"
        files = sorted(cell_dir.glob(pat))
        files = [p for p in files if p.is_file()]
        if not files:
            _warn_missing(
                backend_label,
                f"phase_timing_{phase}",
                cell_dir / pat,
            )
            continue
        df = pd.read_csv(files[0])
        if len(df) != 1:
            warnings.warn(
                f"[{backend_label}] {files[0]}: expected 1 summary row, got {len(df)}",
                UserWarning,
                stacklevel=2,
            )
        row = df.iloc[0]
        out[phase] = _mean_ms_from_row(row, phase)

    return out if out else None


def plot_resource_util_overlay(
    series: list[tuple[str, pd.DataFrame]],
    out: Path,
    *,
    title: str,
    baseline_label: str | None = None,
) -> None:
    if not series:
        warnings.warn("Skipping resource_util figure: no backends with data", UserWarning)
        return
    y_cols: list[str] | None = None
    for label, df in series:
        mc = _metric_columns(df)
        if not mc:
            continue
        y_cols = mc
        break
    if not y_cols:
        warnings.warn("Skipping resource_util figure: no metric columns", UserWarning)
        return

    n = len(y_cols)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 2.4 * n), sharex=True, layout="tight")
    if n == 1:
        axes = [axes]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    base_df: pd.DataFrame | None = None
    if baseline_label is not None:
        for lab, df in series:
            if lab == baseline_label:
                base_df = df
                break
        if base_df is None:
            warnings.warn(
                f"Resource util: baseline {baseline_label!r} not in series; plotting raw values",
                UserWarning,
                stacklevel=2,
            )

    for ax, col in zip(axes, y_cols):
        ymax = 0.0
        use_ratio = (
            baseline_label is not None
            and base_df is not None
            and col in base_df.columns
            and "step" in base_df.columns
        )
        if use_ratio:
            base_sub = base_df[["step", col]].copy()
            base_sub[col] = pd.to_numeric(base_sub[col], errors="coerce")
            base_sub = base_sub.rename(columns={col: "_base"})
        for i, (blab, df) in enumerate(series):
            if col not in df.columns or "step" not in df.columns:
                continue
            c = colors[i % len(colors)]
            if use_ratio:
                m = df[["step", col]].merge(base_sub, on="step", how="inner")
                y = pd.to_numeric(m[col], errors="coerce")
                yb = pd.to_numeric(m["_base"], errors="coerce")
                ratio = (y / yb).where((yb != 0) & np.isfinite(yb))
                step = m["step"]
                ax.plot(step, ratio, linewidth=1.0, label=blab, color=c)
                if ratio.notna().any():
                    ymax = max(ymax, float(ratio.max(skipna=True)))
            else:
                step = df["step"]
                y = pd.to_numeric(df[col], errors="coerce")
                ax.plot(step, y, linewidth=1.0, label=blab, color=c)
                if y.notna().any():
                    ymax = max(ymax, float(y.max(skipna=True)))
        if use_ratio:
            ax.set_ylabel(f"{col}\n(ratio vs {baseline_label})")
            ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--", alpha=0.75)
            if ymax > 0:
                ax.set_ylim(0, max(ymax * 1.08, 1.02))
            else:
                ax.set_ylim(0, 1)
        else:
            ax.set_ylabel(col)
            if ymax > 0:
                ax.set_ylim(0, ymax * 1.08)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("step")
    last_step = series[0][1]["step"]
    _set_xlim_step_from_zero(axes[-1], last_step)
    fig.suptitle(title, fontsize=11, y=1.002)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_phase_timing_bars(
    by_backend: dict[str, dict[str, float]],
    out: Path,
    *,
    title: str,
    baseline_label: str | None = None,
) -> None:
    if not by_backend:
        warnings.warn("Skipping phase_timing figure: no backends with data", UserWarning)
        return
    labels = list(by_backend.keys())
    # Only phases that appear for at least one backend with finite value
    phases_present: list[tuple[str, str, str]] = []
    for phase, plab, col in PHASE_SPEC:
        if any(
            np.isfinite(by_backend[b].get(phase, float("nan")))
            for b in labels
        ):
            phases_present.append((phase, plab, col))
    if not phases_present:
        warnings.warn("Skipping phase_timing figure: no finite phase values", UserWarning)
        return
    if len(phases_present) < 3:
        warnings.warn(
            f"Phase timing: only {len(phases_present)} phase(s) present (not all three); "
            "missing phases were not measured, not zero.",
            UserWarning,
        )

    base: dict[str, float] | None = None
    if baseline_label is not None and baseline_label in by_backend:
        base = by_backend[baseline_label]
        if not all(
            np.isfinite(base.get(pk, float("nan"))) and base.get(pk, float("nan")) != 0
            for pk, _, _ in phases_present
        ):
            warnings.warn(
                f"Phase timing: baseline {baseline_label!r} lacks a finite non-zero value for "
                "every plotted phase; plotting raw ms",
                UserWarning,
                stacklevel=2,
            )
            base = None
    elif baseline_label is not None:
        warnings.warn(
            f"Phase timing: baseline {baseline_label!r} missing; plotting raw ms",
            UserWarning,
            stacklevel=2,
        )

    x = np.arange(len(labels), dtype=float)
    n_phases = len(phases_present)
    width = min(0.22, 0.8 / (n_phases + 1))
    fig, ax = plt.subplots(figsize=(max(5.0, 1.3 * len(labels)), 5.0), layout="constrained")
    for i, (phase_key, _plab, color) in enumerate(phases_present):
        heights = [by_backend[b].get(phase_key, float("nan")) for b in labels]
        if base is not None:
            vb = base.get(phase_key, float("nan"))
            heights = [h / vb if np.isfinite(h) else float("nan") for h in heights]
        pos = x + (i - (n_phases - 1) / 2.0) * width
        ax.bar(
            pos,
            heights,
            width * 0.92,
            label=next(pl for pk, pl, _ in PHASE_SPEC if pk == phase_key),
            color=color,
            alpha=0.92,
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Storage backend", fontsize=11)
    if base is not None:
        ax.set_ylabel(f"Mean time (ratio vs {baseline_label})", fontsize=11)
        ax.axhline(1.0, color="0.35", linewidth=0.9, linestyle="--", alpha=0.75, zorder=0)
    else:
        ax.set_ylabel("Mean time (ms)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(title="Phase", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_phase_timing_totals(
    by_backend: dict[str, dict[str, float]],
    out: Path,
    *,
    title: str,
    baseline_label: str | None = None,
) -> None:
    if not by_backend:
        return
    labels = list(by_backend.keys())
    totals = []
    for b in labels:
        s = 0.0
        for phase, _pl, _c in PHASE_SPEC:
            v = by_backend[b].get(phase, float("nan"))
            if np.isfinite(v):
                s += v
        totals.append(s)
    if not any(t > 0 for t in totals if np.isfinite(t)):
        return
    yvals = list(totals)
    ylab = "Sum of phase means (ms)"
    ratio_ok = False
    if baseline_label is not None:
        yvals, ratio_ok = _ratio_vs_baseline(
            labels, yvals, baseline_label, context="Phase timing totals"
        )
        if ratio_ok:
            ylab = f"Sum of phase means (ratio vs {baseline_label})"
    fig, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(labels)), 4.0), layout="constrained")
    ax.bar(labels, yvals, color="#5c4b7a", alpha=0.9)
    if ratio_ok:
        ax.axhline(1.0, color="0.35", linewidth=0.9, linestyle="--", alpha=0.75, zorder=0)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_xlabel("Storage backend", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_simple_bars(
    labels: list[str],
    values: list[float],
    ylabel: str,
    out: Path,
    *,
    title: str,
    baseline_label: str | None = None,
) -> None:
    yplot = list(values)
    ylab = ylabel
    ratio_ok = False
    if baseline_label is not None:
        yplot, ratio_ok = _ratio_vs_baseline(
            labels, yplot, baseline_label, context="Train duration"
        )
        if ratio_ok:
            ylab = f"{ylabel} (ratio vs {baseline_label})"
    fig, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(labels)), 4.0), layout="constrained")
    ax.bar(labels, yplot, color="#2b8cbe", alpha=0.9)
    if ratio_ok:
        ax.axhline(1.0, color="0.35", linewidth=0.9, linestyle="--", alpha=0.75, zorder=0)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_xlabel("Storage backend", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_multi_metric_bars(
    labels: list[str],
    metrics: list[tuple[str, str, list[float]]],
    out: Path,
    *,
    title: str,
    baseline_label: str | None = None,
) -> None:
    """metrics: (key, ylabel_suffix, per-backend values)."""
    if not labels or not metrics:
        return
    n = len(metrics)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(max(4.0, 1.2 * len(labels)), 2.8 * n), layout="constrained")
    if n == 1:
        axes = [axes]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (key, ylab, vals), c in zip(axes, metrics, colors):
        yplot = list(vals)
        ylab_plot = ylab
        ratio_ok = False
        if baseline_label is not None:
            yplot, ratio_ok = _ratio_vs_baseline(
                labels, yplot, baseline_label, context=f"CodeCarbon ({key})"
            )
            if ratio_ok:
                ylab_plot = f"{ylab} (ratio vs {baseline_label})"
        ax.bar(labels, yplot, color=c, alpha=0.88)
        if ratio_ok:
            ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--", alpha=0.75, zorder=0)
        ax.set_ylabel(ylab_plot, fontsize=10)
        ax.set_title(key, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle(title, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _codecarbon_scalars(path: Path) -> dict[str, float] | None:
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    row = df.iloc[0]
    out = {}
    for k in ("duration", "energy_consumed", "emissions"):
        if k in row.index and pd.notna(row[k]):
            out[k] = float(row[k])
    return out if out else None


def _scan_roots(parent: Path) -> list[tuple[Path, str]]:
    parent = parent.resolve()
    if not parent.is_dir():
        raise SystemExit(f"Not a directory: {parent}")
    pairs: list[tuple[Path, str]] = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("data_"):
            continue
        label = SCAN_LABEL_ALIASES.get(name, name.removeprefix("data_") or name)
        pairs.append((child.resolve(), label))
    if not pairs:
        raise SystemExit(
            f"No subdirectories matching data_* under {parent}"
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True, metavar="N")
    parser.add_argument("--worker", type=int, required=True, metavar="M")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: results/plots/compare_data_types/batch_N_worker_M)",
    )
    parser.add_argument(
        "--scan-parent",
        type=Path,
        default=None,
        metavar="DIR",
        help="Use immediate data_* child dirs of DIR as roots with default labels",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="PATH",
        help="Backend results root (repeatable); pair with --label",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="NAME",
        help="Label for preceding --root (repeatable)",
    )
    parser.add_argument(
        "--normalize-to",
        type=str,
        default="memory",
        metavar="LABEL",
        help="Plot ratios vs this backend (its bars/lines = 1). Use empty string for raw units.",
    )
    args = parser.parse_args()

    if args.scan_parent is not None:
        if args.root:
            raise SystemExit("Use either --scan-parent or --root/--label, not both")
        backends = _scan_roots(args.scan_parent)
    else:
        if len(args.root) != len(args.label) or not args.root:
            raise SystemExit("Provide --scan-parent DIR or equal numbers of --root and --label")
        backends = [(Path(r).resolve(), lab) for r, lab in zip(args.root, args.label)]

    batch, worker = args.batch, args.worker
    if args.out_dir is None:
        out_dir = Path("results/plots/compare_data_types") / f"batch_{batch}_worker_{worker}"
    else:
        out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = args.normalize_to.strip() or None

    ru_series: list[tuple[str, pd.DataFrame]] = []
    duration_ms: list[tuple[str, float]] = []
    phase_rows: dict[str, dict[str, float]] = {}
    coarse_vals: dict[str, dict[str, float]] = {}
    e2e_vals: dict[str, dict[str, float]] = {}

    extra_run_patterns = [
        "resource_util_run_*.csv",
        "memory_resource_util_run_*.csv",
        "*_phase_timing_*_summary.csv",
        "run_*_cc_full_coarse_rank_*.csv",
        "memory_run_*_cc_full_coarse_rank_*.csv",
    ]

    for root, blabel in backends:
        cell = _cell_dir(root, batch, worker)
        if not cell.is_dir():
            _warn_missing(blabel, "batch/worker cell directory", cell)
            continue

        _warn_extra_runs(cell, blabel, extra_run_patterns)

        ru_path = _resolve_resource_util_csv(cell, blabel)
        if ru_path is not None:
            df = pd.read_csv(ru_path)
            ru_series.append((blabel, df))

        dur_path = _resolve_train_duration_txt(cell, blabel)
        if dur_path is not None:
            ms = _parse_duration_ms(dur_path)
            if ms is None:
                warnings.warn(
                    f"[{blabel}] could not parse duration_ms from {dur_path}",
                    UserWarning,
                )
            else:
                duration_ms.append((blabel, ms))

        pr = _collect_phase_timing_row(cell, blabel)
        if pr:
            phase_rows[blabel] = pr

        mem = _uses_memory_prefix(blabel)
        cc_glob = (
            f"{RUN_TOKEN_MEM}cc_full_coarse_rank_*.csv"
            if mem
            else f"{RUN_TOKEN_NONMEM}cc_full_coarse_rank_*.csv"
        )
        cc_path = _glob_single(cell, blabel, cc_glob, "cc_full_coarse")
        if cc_path is not None:
            scal = _codecarbon_scalars(cc_path)
            if scal:
                coarse_vals[blabel] = scal

        e2e_glob = (
            f"{RUN_TOKEN_MEM}cc_e2e_full_rank_*.csv"
            if mem
            else f"{RUN_TOKEN_NONMEM}cc_e2e_full_rank_*.csv"
        )
        e2e_path = _glob_single(cell, blabel, e2e_glob, "cc_e2e_full")
        if e2e_path is not None:
            scal = _codecarbon_scalars(e2e_path)
            if scal:
                e2e_vals[blabel] = scal

    t_note = f", vs {baseline}" if baseline else ""
    plot_resource_util_overlay(
        ru_series,
        out_dir / f"compare_resource_util_run_{RUN_INDEX}.png",
        title=f"Resource util (run {RUN_INDEX}) batch={batch} worker={worker}{t_note}",
        baseline_label=baseline,
    )

    plot_phase_timing_bars(
        phase_rows,
        out_dir / f"compare_phase_timing_run_{RUN_INDEX}.png",
        title=f"Phase timing mean (run {RUN_INDEX}) batch={batch} worker={worker}{t_note}",
        baseline_label=baseline,
    )
    plot_phase_timing_totals(
        phase_rows,
        out_dir / f"compare_phase_timing_total_run_{RUN_INDEX}.png",
        title=f"Sum of phase means (run {RUN_INDEX}) batch={batch} worker={worker}{t_note}",
        baseline_label=baseline,
    )

    if duration_ms:
        plot_simple_bars(
            [t[0] for t in duration_ms],
            [t[1] for t in duration_ms],
            "Train duration (ms)",
            out_dir / f"compare_train_duration_run_{RUN_INDEX}.png",
            title=f"Resource util wall time (run {RUN_INDEX}) batch={batch} worker={worker}{t_note}",
            baseline_label=baseline,
        )
    else:
        warnings.warn("Skipping train_duration figure: no data", UserWarning)

    if coarse_vals:
        labels = list(coarse_vals.keys())
        metrics = []
        for key, ylab in (
            ("duration", "Duration (s)"),
            ("energy_consumed", "Energy (kWh)"),
            ("emissions", "Emissions (kg CO2eq)"),
        ):
            vals = [coarse_vals[b].get(key, float("nan")) for b in labels]
            if any(np.isfinite(v) for v in vals):
                metrics.append((key, ylab, vals))
        if metrics:
            plot_multi_metric_bars(
                labels,
                metrics,
                out_dir / f"compare_codecarbon_coarse_run_{RUN_INDEX}.png",
                title=f"CodeCarbon full coarse (run {RUN_INDEX}) batch={batch} worker={worker}{t_note}",
                baseline_label=baseline,
            )
    else:
        warnings.warn("Skipping codecarbon coarse figure: no data", UserWarning)

    if e2e_vals:
        labels = list(e2e_vals.keys())
        metrics = []
        if any(np.isfinite(e2e_vals[b].get("energy_consumed", float("nan"))) for b in labels):
            metrics.append(
                (
                    "energy_consumed",
                    "Energy (kWh)",
                    [e2e_vals[b].get("energy_consumed", float("nan")) for b in labels],
                )
            )
        if any(np.isfinite(e2e_vals[b].get("emissions", float("nan"))) for b in labels):
            metrics.append(
                (
                    "emissions",
                    "Emissions (kg CO2eq)",
                    [e2e_vals[b].get("emissions", float("nan")) for b in labels],
                )
            )
        if metrics:
            plot_multi_metric_bars(
                labels,
                metrics,
                out_dir / f"compare_codecarbon_e2e_energy_run_{RUN_INDEX}.png",
                title=f"CodeCarbon e2e full run (run {RUN_INDEX}) batch={batch} worker={worker}{t_note}",
                baseline_label=baseline,
            )
    else:
        warnings.warn("Skipping codecarbon e2e figure: no data", UserWarning)


if __name__ == "__main__":
    main()
