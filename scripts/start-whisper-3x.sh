#!/bin/bash
# Same as start-whisper.sh, but runs it multiple times (default: 3).
# Each run gets WHISPER_CODECARBON_RUN_NUM=1,2,... (CodeCarbon filenames) and
# RUN_REPEAT_INDEX=1,2,... (resource_util: <stem>_run_N.csv). Forward all flags
# to start-whisper.sh.
#
# Job-level repeats (this script): WHISPER_REPEAT_COUNT (default 3). Dataset
# virtual length uses --repeat / WHISPER_REPEAT (see start-whisper.sh).
#
# Override count: WHISPER_REPEAT_COUNT=5 ./scripts/start-whisper-3x.sh ...
#
# Implementation: scripts/run_repeat.sh
#
# --- Copy-paste template (flags optional; defaults match start-whisper.sh) ---
#
# ./scripts/start-whisper-3x.sh \
#   --batch_size 4 \
#   --learning_rate 1e-6 \
#   --num_workers 0 \
#   --trainer_stats codecarbon \
#   (default output_dir: results/data/batch_<batch>_worker_<workers>)
#   --repeat 4
#   # default memory_only=0 (on-disk); use --memory_only 1 for RAM-only
# (Do not pass --trainer_stats_configs.codecarbon.run_num: this script sets
#  WHISPER_CODECARBON_RUN_NUM to 1..WHISPER_REPEAT_COUNT each run.)
#
# phase_timing example (writes run_*_phase_timing_*_per_step.csv and *_summary.csv):
#
# ./scripts/start-whisper-3x.sh \
#   --trainer_stats phase_timing
#   # same run_num / output_dir wiring as codecarbon (WHISPER_CODECARBON_RUN_NUM per repeat).
#
# resource_util example:
#
# ./scripts/start-whisper-3x.sh \
#   --batch_size 4 \
#   --learning_rate 1e-6 \
#   --num_workers 2 \
#   --trainer_stats resource_util \
#   (default output_dir: results/data/batch_<batch>_worker_<workers>)
#   --stats_output_file resource_util.csv \
#   --repeat 4
#   # default memory_only=0 (on-disk); use --memory_only 1 for RAM-only
#
# Env equivalents: WHISPER_BATCH_SIZE, WHISPER_LEARNING_RATE, WHISPER_NUM_WORKERS,
# WHISPER_TRAINER_STATS, WHISPER_CODECARBON_RUN_NUM, WHISPER_OUTPUT_DIR,
# WHISPER_STATS_OUTPUT_FILE, WHISPER_REPEAT, WHISPER_MEMORY_ONLY, WHISPER_DATA_TYPE, etc.
# -------------------------------------------------------------------------------

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
N="${WHISPER_REPEAT_COUNT:-3}"

exec "${SCRIPTS_DIR}/run_repeat.sh" -e WHISPER_CODECARBON_RUN_NUM "$N" "${SCRIPTS_DIR}/start-whisper.sh" "$@"
