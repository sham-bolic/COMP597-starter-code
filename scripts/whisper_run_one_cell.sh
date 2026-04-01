#!/bin/bash
# One Whisper training run on the current machine (no srun): same CLI as start-whisper.sh
set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
# shellcheck source=whisper_srun_hyperparams.sh
. "${SCRIPTS_DIR}/whisper_srun_hyperparams.sh"
# shellcheck source=whisper_launch_cell_body.sh
. "${SCRIPTS_DIR}/whisper_launch_cell_body.sh"

WHISPER_DEFAULT_TRAINER_STATS=codecarbon
whisper_parse_hyperparams "$@"
whisper_apply_run_repeat_resource_util_name
export WHISPER_OUTPUT_DIR

whisper_build_training_args
exec "${SCRIPTS_DIR}/launch.sh" "${ARGS[@]}" "${WHISPER_REST[@]}"
