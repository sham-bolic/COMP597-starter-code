#!/bin/bash
# Same as start-whisper.sh, but runs it multiple times (default: 3).
# Each run gets WHISPER_CODECARBON_RUN_NUM=1,2,... (CodeCarbon filenames) and
# RUN_REPEAT_INDEX=1,2,... (resource_util: <stem>_run_N.csv). Forward all flags
# to start-whisper.sh.
#
# Override count: WHISPER_REPEAT_COUNT=5 ./scripts/start-whisper-3x.sh ...
#
# Implementation: scripts/run_repeat.sh

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
N="${WHISPER_REPEAT_COUNT:-3}"

exec "${SCRIPTS_DIR}/run_repeat.sh" -e WHISPER_CODECARBON_RUN_NUM "$N" "${SCRIPTS_DIR}/start-whisper.sh" "$@"
