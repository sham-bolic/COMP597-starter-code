#!/usr/bin/env bash
# Generate synthetic_whisper on-disk cache (chunks / shard / memmap) on the current host.
# Intended to run inside a Slurm allocation via generate_synthetic_whisper_data_srun.sh
# (same pattern as synthetic_whisper_clean.sh).
#
# Sources config/default_job_config.sh so COMP597_JOB_STUDENT_STORAGE_DIR matches training jobs.
#
# Usage (from repo root, typically via generate_synthetic_whisper_data_srun.sh):
#   ./scripts/generate_synthetic_whisper_data_job.sh --data_type chunks
#   ./scripts/generate_synthetic_whisper_data_job.sh --data_type shard --num_shards 40
#   ./scripts/generate_synthetic_whisper_data_job.sh --data_type memmap
#   ./scripts/generate_synthetic_whisper_data_job.sh --no-force-regenerate --data_type chunks

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

exec python -m src.data.synthetic_whisper.generate "$@"
