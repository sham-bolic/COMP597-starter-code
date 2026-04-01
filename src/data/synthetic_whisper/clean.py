"""CLI to remove synthetic_whisper on-disk caches (manifest + sidecar dirs).

Uses ``clear_synthetic_whisper_disk_cache`` — the same removal path as
``force_regenerate`` before regenerating (see ``data.py``).
"""

from __future__ import annotations

import argparse
import os
import sys

from src.data.synthetic_whisper.data import clear_synthetic_whisper_disk_cache


def _default_data_path() -> str:
    base = os.environ.get("COMP597_JOB_STUDENT_STORAGE_DIR", "").strip()
    if base:
        return os.path.join(base, "synthetic_whisper_data.pt")
    return ""


def _resolve_manifest_path(path: str) -> str:
    """Expand ``${VAR}`` / ``$VAR`` then absolutize (matches evaluated job args)."""
    path = os.path.expandvars(path)
    return os.path.abspath(os.path.expanduser(path))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Manifest .pt path (default: $COMP597_JOB_STUDENT_STORAGE_DIR/synthetic_whisper_data.pt if set)",
    )
    p.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print paths that would be removed without deleting",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Do not prompt for confirmation",
    )
    args = p.parse_args(argv)
    path = args.data_path or _default_data_path()
    if not path:
        print(
            "synthetic_whisper_clean: source config/default_job_config.sh (use scripts/synthetic_whisper_clean.sh) "
            "or set COMP597_JOB_STUDENT_STORAGE_DIR / pass data_path",
            file=sys.stderr,
        )
        return 2
    path = _resolve_manifest_path(path)
    if not args.yes and not args.dry_run:
        try:
            ans = input(f"Remove synthetic_whisper cache for {path!r}? [y/N] ")
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1
    if not args.dry_run:
        print(
            "=============================================================\n"
            "synthetic_whisper: removing on-disk cache (same as force_regenerate)\n"
            "=============================================================",
            flush=True,
        )
    removed = clear_synthetic_whisper_disk_cache(path, dry_run=args.dry_run)
    for r in removed:
        print(f"{'would remove' if args.dry_run else 'removed'}: {r}")
    if not removed:
        print("(nothing to remove)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
