#!/bin/bash
# Single entry point for Whisper + synthetic_whisper on Slurm.
# Learning rate defaults to 1e-6; override with --learning_rate or WHISPER_LEARNING_RATE.
# Default trainer stats: codecarbon. For resource-util CSV logging:
#   ./start-whisper.sh --trainer_stats resource_util

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
# shellcheck source=whisper_srun_hyperparams.sh
. "${SCRIPTS_DIR}/whisper_srun_hyperparams.sh"

WHISPER_DEFAULT_TRAINER_STATS=codecarbon
whisper_parse_hyperparams "$@"

# Literal for expansion on the Slurm job node (see job.sh / default_job_config.sh).
DATA_PATH_LITERAL='${COMP597_JOB_STUDENT_STORAGE_DIR}/synthetic_whisper_data.pt'

ARGS=(
	--logging.level INFO
	--model whisper
	--trainer simple
	--batch_size "${WHISPER_BATCH_SIZE}"
	--learning_rate "${WHISPER_LEARNING_RATE}"
	--data synthetic_whisper
	--data_configs.synthetic_whisper.data_path "${DATA_PATH_LITERAL}"
	--data_configs.synthetic_whisper.num_workers "${WHISPER_NUM_WORKERS}"
	--trainer_stats "${WHISPER_TRAINER_STATS}"
)

# Same WHISPER_OUTPUT_DIR for codecarbon and resource_util; backend-specific paths
# only differ by filename (resource_util CSV) vs CodeCarbon’s own file naming.
case "${WHISPER_TRAINER_STATS}" in
	codecarbon)
		ARGS+=(
			--trainer_stats_configs.codecarbon.run_num "${WHISPER_CODECARBON_RUN_NUM}"
			--trainer_stats_configs.codecarbon.project_name whisper
			--trainer_stats_configs.codecarbon.output_dir "${WHISPER_OUTPUT_DIR}"
		)
		;;
	resource_util)
		ARGS+=(
			--trainer_stats_configs.resource_util.output_dir "${WHISPER_OUTPUT_DIR}"
			--trainer_stats_configs.resource_util.output_file "${WHISPER_STATS_OUTPUT_FILE}"
		)
		;;
esac

"${SCRIPTS_DIR}/srun.sh" "${ARGS[@]}" "${WHISPER_REST[@]}"
