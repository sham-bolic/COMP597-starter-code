"""CLI to generate synthetic_whisper on-disk datasets (chunks / shard / memmap) without training.

Uses the same generation and cache-clearing paths as ``load_data`` + ``force_regenerate`` in
``data.py``. Default behavior clears any existing manifest and sidecar dirs, then rebuilds.
"""

from __future__ import annotations

import argparse
import os
import sys

from src.data.synthetic_whisper.data import (
    SAMPLE_SIZE_DEFAULT,
    generate_memmap,
    generate_samples,
    generate_sharded,
)


def _default_data_path() -> str:
    base = os.environ.get("COMP597_JOB_STUDENT_STORAGE_DIR", "").strip()
    if base:
        return os.path.join(base, "synthetic_whisper_data.pt")
    return ""


def _resolve_manifest_path(path: str) -> str:
    path = os.path.expandvars(path)
    return os.path.abspath(os.path.expanduser(path))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_type",
        required=True,
        choices=("chunks", "shard", "memmap"),
        help="Disk-backed layout to generate (``memory`` is not supported; no on-disk cache).",
    )
    p.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Manifest .pt path (default: $COMP597_JOB_STUDENT_STORAGE_DIR/synthetic_whisper_data.pt if set)",
    )
    p.add_argument(
        "--num_unique_samples",
        type=int,
        default=int(os.environ.get("WHISPER_NUM_UNIQUE_SAMPLES", SAMPLE_SIZE_DEFAULT)),
        help=f"Pool size N (default: env WHISPER_NUM_UNIQUE_SAMPLES or {SAMPLE_SIZE_DEFAULT})",
    )
    p.add_argument("--num_labels", type=int, default=10)
    p.add_argument(
        "--chunk_size",
        type=int,
        default=int(os.environ.get("WHISPER_CHUNK_SIZE", 200)),
        help="Samples per chunk file for data_type=chunks (default: 200 or WHISPER_CHUNK_SIZE)",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=int(os.environ.get("WHISPER_NUM_SHARDS", 40)),
        help="Shard count for data_type=shard (default: 40 or WHISPER_NUM_SHARDS)",
    )
    p.add_argument(
        "--no-force-regenerate",
        action="store_true",
        help="If the manifest already exists, skip generation (no clean, no rebuild).",
    )
    args = p.parse_args(argv)

    path = args.data_path or _default_data_path()
    if not path:
        print(
            "synthetic_whisper_generate: set COMP597_JOB_STUDENT_STORAGE_DIR or pass data_path "
            "(see scripts/generate_synthetic_whisper_data_job.sh)",
            file=sys.stderr,
        )
        return 2
    path = _resolve_manifest_path(path)

    n = max(1, int(args.num_unique_samples))
    num_labels = max(1, int(args.num_labels))

    if args.no_force_regenerate and os.path.isfile(path):
        print(f"synthetic_whisper_generate: manifest exists, skipping (--no-force-regenerate): {path}", flush=True)
        return 0

    print(
        "=============================================================\n"
        f"synthetic_whisper: generating data_type={args.data_type} n={n} manifest={path}\n"
        "=============================================================",
        flush=True,
    )

    if args.data_type == "chunks":
        generate_samples(n, path, num_labels, max(1, int(args.chunk_size)))
    elif args.data_type == "shard":
        generate_sharded(n, path, num_labels, max(1, int(args.num_shards)))
    else:
        generate_memmap(n, path, num_labels)

    print("synthetic_whisper_generate: done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
