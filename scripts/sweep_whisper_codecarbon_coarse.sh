#!/usr/bin/env bash
# Sweep Whisper with codecarbon_coarse; artifacts under results/data_cc_coarse/ by default
# (override with SWEEP_DATA_ROOT). Writes:
#   run_*_cc_full_coarse_rank_*.csv
#   run_*_cc_step_coarse_rank_*-coarse_steps.csv
#   run_*_cc_coarse_run_summary_rank_*.csv  (total_train_wall_s + per-step estimates from chunks)
#
# Usage:
#   ./scripts/sweep_whisper_codecarbon_coarse.sh
#   BATCH_SIZES=4,8 WORKER_SIZES=0,2 ./scripts/sweep_whisper_codecarbon_coarse.sh
#   SWEEP_DATA_ROOT=results/my_cc_coarse ./scripts/sweep_whisper_codecarbon_coarse.sh -- --repeat 2
#   WHISPER_CODECARBON_COARSE_STEP_INTERVAL=20 ./scripts/sweep_whisper_codecarbon_coarse.sh

set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
ROOT="${SWEEP_DATA_ROOT:-results/data_cc_coarse}"
export WHISPER_SWEEP_DATA_ROOT="$ROOT"

BATCH_SIZES=${BATCH_SIZES:-"4,8,16"}
WORKER_SIZES=${WORKER_SIZES:-"0"}

exec "${SCRIPTS_DIR}/sweep_whisper_batch_workers.sh" \
	--trainer_stats codecarbon_coarse \
	--batch_sizes "$BATCH_SIZES" \
	--worker_sizes "$WORKER_SIZES" \
	-- "$@"
