#!/usr/bin/env python3
"""Plot each metric column in a resource_util CSV against ``step`` (x-axis).

Default output layout: ``results/plots/<path_under_results>/<stem>_all_metrics.png``
  (e.g. ``results/data/test/resource_util.csv`` → ``results/plots/data/test/...``).

With ``--average``, ``input_path`` must be a directory containing
``<run_stem>_run_1.csv``, ``<run_stem>_run_2.csv``, ... (e.g. from ``start-whisper-3x``).
Rows are aligned on ``step`` and metric columns are averaged across runs.

Example:
  python3 scripts/plot_all_resource_util.py results/data/test/resource_util.csv
  python3 scripts/plot_all_resource_util.py results/data/test/resource_util.csv -o /tmp/custom.png
  python3 scripts/plot_all_resource_util.py run.csv --separate
  python3 scripts/plot_all_resource_util.py --average results/data/batch_4_worker_0
  python3 scripts/plot_all_resource_util.py --average results/foo --run-stem mytrace
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR_NAME = "results"
DEFAULT_PLOTS_ROOT = Path("results") / "plots"


def _mirror_plots_subdir(csv_path: Path, plots_root: Path) -> Path:
    """Map ``results/.../dir/file.csv`` → ``plots_root/.../dir``; else ``plots_root/<parent-name>``."""
    csv_path = csv_path.resolve()
    parts = csv_path.parts
    try:
        i = parts.index(RESULTS_DIR_NAME)
    except ValueError:
        rel = Path(csv_path.parent.name)
    else:
        inner = parts[i + 1 : -1]
        rel = Path(*inner) if inner else Path()
    return (plots_root.resolve() / rel)


def _metric_columns(df: pd.DataFrame) -> list[str]:
    if "step" not in df.columns:
        raise SystemExit("CSV must contain a 'step' column.")
    return [c for c in df.columns if c != "step"]


def _run_csv_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"_run_(\d+)\.csv$", path.name, re.IGNORECASE)
    n = int(m.group(1)) if m else 0
    return (n, path.name)


def average_run_csvs_in_dir(directory: Path, run_stem: str) -> tuple[pd.DataFrame, list[Path]]:
    """Load ``{run_stem}_run_N.csv`` files and return mean of numeric columns by ``step``."""
    directory = directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")
    pattern = f"{run_stem}_run_*.csv"
    files = sorted(directory.glob(pattern), key=_run_csv_sort_key)
    if not files:
        raise SystemExit(f"No files matching {pattern!r} in {directory}")
    dfs = [pd.read_csv(f) for f in files]
    for f, df in zip(files, dfs):
        if "step" not in df.columns:
            raise SystemExit(f"Missing 'step' column in {f}")
    combined = pd.concat(dfs, ignore_index=True)
    mean_df = combined.groupby("step", as_index=False).mean(numeric_only=True)
    mean_df = mean_df.sort_values("step").reset_index(drop=True)
    return mean_df, files


def _set_xlim_step_from_zero(ax, step: pd.Series, *, right_pad_ratio: float = 0.02) -> None:
    """x-axis (step) starts at 0 with no left margin; small padding on the right only."""
    xs = pd.to_numeric(step, errors="coerce")
    if not xs.notna().any():
        ax.set_xlim(0, 1)
        return
    xmax = float(xs.max(skipna=True))
    if xmax <= 0:
        ax.set_xlim(0, 1)
    else:
        ax.set_xlim(0, xmax * (1.0 + right_pad_ratio))


def _set_ylim_bottom_zero_with_headroom(ax, y: pd.Series, *, pad_ratio: float = 0.08) -> None:
    """y-axis from 0 with relative padding above max so the trace is not clipped."""
    vals = pd.to_numeric(y, errors="coerce")
    if not vals.notna().any():
        ax.set_ylim(0, 1)
        return
    ymax = float(vals.max(skipna=True))
    if ymax <= 0:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, ymax * (1.0 + pad_ratio))


def _filter_columns(
    all_cols: list[str],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    """Apply --metrics / --exclude-metrics filters. Patterns are case-insensitive substrings."""
    cols = list(all_cols)
    if include:
        pats = [p.lower() for p in include]
        cols = [c for c in cols if any(p in c.lower() for p in pats)]
    if exclude:
        pats = [p.lower() for p in exclude]
        cols = [c for c in cols if not any(p in c.lower() for p in pats)]
    return cols


def plot_combined(
    df: pd.DataFrame,
    title: str,
    path: Path,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> None:
    y_cols = _filter_columns(_metric_columns(df), include, exclude)
    if not y_cols:
        raise SystemExit("No metric columns to plot (after filtering).")

    n = len(y_cols)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 2.4 * n), sharex=True, layout="tight")
    if n == 1:
        axes = [axes]
    step = df["step"]
    for ax, col in zip(axes, y_cols):
        ax.plot(step, pd.to_numeric(df[col], errors="coerce"), linewidth=1.0)
        ax.set_ylabel(col)
        _set_ylim_bottom_zero_with_headroom(ax, df[col])
        _set_xlim_step_from_zero(ax, step)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=True)
    fig.suptitle(title, fontsize=11, y=1.002)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_compare(
    csv_paths: list[Path],
    labels: list[str],
    title: str,
    path: Path,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> None:
    """Overlay the same metric(s) from multiple CSVs on shared axes."""
    dfs = []
    for p in csv_paths:
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")
        dfs.append(pd.read_csv(p))

    common_cols: list[str] | None = None
    for df in dfs:
        filtered = _filter_columns(_metric_columns(df), include, exclude)
        if common_cols is None:
            common_cols = filtered
        else:
            common_cols = [c for c in common_cols if c in filtered]
    if not common_cols:
        raise SystemExit("No common metric columns across CSVs (after filtering).")

    n = len(common_cols)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 3.0 * n), sharex=True, layout="tight")
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, common_cols):
        all_y = pd.Series(dtype=float)
        all_x = pd.Series(dtype=float)
        for i, (df, label) in enumerate(zip(dfs, labels)):
            step = df["step"]
            y = pd.to_numeric(df[col], errors="coerce")
            # Alternate solid vs dotted when inputs are ordered plain, r1, plain, r1, ...
            ls = "-" if i % 2 == 0 else ":"
            ax.plot(step, y, linewidth=1.8, linestyle=ls, label=label)
            all_y = pd.concat([all_y, y], ignore_index=True)
            all_x = pd.concat([all_x, step], ignore_index=True)
        ax.set_ylabel(col)
        _set_ylim_bottom_zero_with_headroom(ax, all_y)
        _set_xlim_step_from_zero(ax, all_x)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.setp(ax.get_xticklabels(), visible=True)
    fig.suptitle(title, fontsize=11, y=1.002)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_separate(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> None:
    y_cols = _filter_columns(_metric_columns(df), include, exclude)
    if not y_cols:
        raise SystemExit("No metric columns to plot (after filtering).")
    out_dir.mkdir(parents=True, exist_ok=True)
    step = df["step"]
    for col in y_cols:
        fig, ax = plt.subplots(figsize=(8, 3.5), layout="tight")
        ax.plot(step, pd.to_numeric(df[col], errors="coerce"), linewidth=1.0)
        ax.set_xlabel("step")
        ax.set_ylabel(col)
        _set_ylim_bottom_zero_with_headroom(ax, df[col])
        _set_xlim_step_from_zero(ax, step)
        ax.grid(True, alpha=0.3)
        path = out_dir / f"{stem}_{col}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot resource_util.csv metrics vs step (one combined figure, or one PNG per metric)."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="+",
        help="Path(s) to resource_util.csv. Multiple paths enable --compare mode.",
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help=(
            "Average all {run_stem}_run_*.csv in input_path (directory). "
            "Typical layout from start-whisper-3x."
        ),
    )
    parser.add_argument(
        "--run-stem",
        default="resource_util",
        metavar="STEM",
        help="Basename for run files: {stem}_run_N.csv (default: resource_util)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Combined PNG path (default: "
            f"{DEFAULT_PLOTS_ROOT}/<mirror-of-results-path>/"
            "<stem>_all_metrics.png)"
        ),
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=DEFAULT_PLOTS_ROOT,
        help=f"Base directory for outputs (default: {DEFAULT_PLOTS_ROOT})",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Also write one PNG per metric into the same output directory as the combined plot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for --separate plots only (default: parent of combined PNG)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="PAT",
        help="Only plot columns matching these substrings (case-insensitive). E.g. --metrics gpu io",
    )
    parser.add_argument(
        "--exclude-metrics",
        nargs="+",
        default=None,
        metavar="PAT",
        help="Exclude columns matching these substrings (case-insensitive). E.g. --exclude-metrics write logical",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom plot title (default: CSV filename or run-stem summary)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Overlay multiple CSVs on the same axes. Pass multiple input_path args.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Legend labels for --compare mode (one per input CSV).",
    )
    args = parser.parse_args()

    plots_root = args.plots_root
    if not plots_root.is_absolute():
        plots_root = Path.cwd() / plots_root

    if args.compare or len(args.input_path) > 1:
        csv_paths = [p.resolve() for p in args.input_path]
        labels = args.labels or [p.stem for p in csv_paths]
        if len(labels) != len(csv_paths):
            raise SystemExit(f"--labels count ({len(labels)}) must match input count ({len(csv_paths)})")
        plot_title = args.title or "comparison"
        if args.output_dir is not None:
            out_dir = args.output_dir.resolve()
        else:
            out_dir = plots_root / "compare"
        out_path = args.output.resolve() if args.output else out_dir / "compare_all_metrics.png"
        plot_compare(csv_paths, labels, plot_title, out_path, include=args.metrics, exclude=args.exclude_metrics)
        return

    if args.average:
        directory = args.input_path[0].resolve()
        df, run_files = average_run_csvs_in_dir(directory, args.run_stem)
        mirror_ref = run_files[0]
        plot_title = args.title or f"{args.run_stem} (mean of {len(run_files)} runs)"
        out_stem = f"{args.run_stem}_mean"
        out_dir_base = _mirror_plots_subdir(mirror_ref, plots_root)
        combined_out = args.output
        if combined_out is None:
            combined_out = out_dir_base / f"{out_stem}_all_metrics.png"
        else:
            combined_out = combined_out.resolve()
        plot_combined(df, title=plot_title, path=combined_out, include=args.metrics, exclude=args.exclude_metrics)
        if args.separate:
            sep_dir = args.output_dir
            if sep_dir is None:
                sep_dir = combined_out.parent
            else:
                sep_dir = sep_dir.resolve()
            plot_separate(df, sep_dir, stem=out_stem, include=args.metrics, exclude=args.exclude_metrics)
        return

    csv_path = args.input_path[0].resolve()
    if not csv_path.is_file():
        raise SystemExit(f"Not a file: {csv_path}")

    df = pd.read_csv(csv_path)

    if args.output_dir is not None:
        out_dir_base = args.output_dir.resolve()
    else:
        out_dir_base = _mirror_plots_subdir(csv_path, plots_root)

    combined_out = args.output
    if combined_out is None:
        combined_out = out_dir_base / f"{csv_path.stem}_all_metrics.png"
    else:
        combined_out = combined_out.resolve()

    plot_title = args.title or csv_path.name
    plot_combined(df, title=plot_title, path=combined_out, include=args.metrics, exclude=args.exclude_metrics)

    if args.separate:
        plot_separate(df, combined_out.parent, stem=csv_path.stem, include=args.metrics, exclude=args.exclude_metrics)


if __name__ == "__main__":
    main()
