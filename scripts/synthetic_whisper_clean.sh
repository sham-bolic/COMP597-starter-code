#!/usr/bin/env bash
# Remove synthetic_whisper manifest + chunk/shard/memmap sidecar dirs.
# Uses the same cache-clearing logic as load_data + force_regenerate (clear_synthetic_whisper_disk_cache).
# Sources config/default_job_config.sh (+ COMP597_JOB_CONFIG if present) like scripts/job.sh so
# COMP597_JOB_STUDENT_STORAGE_DIR matches Slurm training jobs.
#
# This always runs on the *current* host unless you submit through Slurm.
# On this cluster, bare ``srun`` needs ``--mem`` and the rest; easiest is the **course** wrapper:
#   ./scripts/synthetic_whisper_clean_srun.sh -n
# That sets COMP597_JOB_COMMAND and runs ./scripts/srun.sh (same flags + conda activate as training).
# Manual equivalent from repo root:
#   export COMP597_JOB_COMMAND="$(pwd)/scripts/synthetic_whisper_clean.sh"
#   ./scripts/srun.sh -n
#
# Usage:
#   ./scripts/synthetic_whisper_clean.sh              # default manifest under $COMP597_JOB_STUDENT_STORAGE_DIR
#   ./scripts/synthetic_whisper_clean.sh /path/to/manifest.pt
#   ./scripts/synthetic_whisper_clean.sh -n          # dry-run
#   ./scripts/synthetic_whisper_clean.sh -y /path    # no confirm

set -euo pipefail

ROOT="$(readlink -f "$(dirname "$0")/..")"
cd "$ROOT"

DEFAULT_CONFIG_FILE="${ROOT}/config/default_job_config.sh"
# shellcheck disable=SC1090
. "${DEFAULT_CONFIG_FILE}"
if [[ -f "${COMP597_JOB_CONFIG:-}" ]]; then
	# shellcheck disable=SC1090
	. "${COMP597_JOB_CONFIG}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m src.data.synthetic_whisper.clean "$@"
