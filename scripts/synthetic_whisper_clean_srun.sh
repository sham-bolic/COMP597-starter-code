#!/usr/bin/env bash
# Run synthetic_whisper_clean.sh on a Slurm node via ./scripts/srun.sh (same --mem, account,
# qos, partition, conda activate as training jobs). Avoids bare "srun" missing required flags.
#
# Usage (from repo root, same args as synthetic_whisper_clean.sh):
#   ./scripts/synthetic_whisper_clean_srun.sh -n
#   ./scripts/synthetic_whisper_clean_srun.sh -y
#
# Equivalent manual form:
#   export COMP597_JOB_COMMAND="/path/to/COMP597-starter-code/scripts/synthetic_whisper_clean.sh"
#   ./scripts/srun.sh -n
#
# Optional: shorten the allocation, e.g. COMP597_SLURM_TIME_LIMIT=0:15:00 ./scripts/synthetic_whisper_clean_srun.sh -n

set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")

export COMP597_JOB_COMMAND="${SCRIPTS_DIR}/synthetic_whisper_clean.sh"
cd "${REPO_DIR}"

exec "${SCRIPTS_DIR}/srun.sh" "$@"
