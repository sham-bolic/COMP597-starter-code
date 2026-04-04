#!/usr/bin/env bash
# Generate synthetic_whisper disk data on a Slurm node via ./scripts/srun.sh
# (same --mem, account, qos, partition, conda activate as training jobs).
#
# Usage (from repo root):
#   ./scripts/generate_synthetic_whisper_data_srun.sh --data_type chunks
#   ./scripts/generate_synthetic_whisper_data_srun.sh --data_type shard --num_shards 40
#   ./scripts/generate_synthetic_whisper_data_srun.sh --data_type memmap
#
# Optional: shorten allocation, e.g.
#   COMP597_SLURM_TIME_LIMIT=0:45:00 ./scripts/generate_synthetic_whisper_data_srun.sh --data_type memmap

set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")

export COMP597_JOB_COMMAND="${SCRIPTS_DIR}/generate_synthetic_whisper_data_job.sh"
cd "${REPO_DIR}"

exec "${SCRIPTS_DIR}/srun.sh" "$@"
