# shellcheck shell=bash
# Sourced after whisper_parse_hyperparams sets WHISPER_* (start-whisper.sh, whisper_run_one_cell.sh).

whisper_apply_run_repeat_resource_util_name() {
	if [[ -n "${RUN_REPEAT_INDEX:-}" && "${WHISPER_TRAINER_STATS}" == "resource_util" ]]; then
		local _ru _ru_stem _ru_ext
		_ru="${WHISPER_STATS_OUTPUT_FILE}"
		if [[ "$_ru" == *.* ]]; then
			_ru_stem="${_ru%.*}"
			_ru_ext=".${_ru##*.}"
		else
			_ru_stem="$_ru"
			_ru_ext=""
		fi
		WHISPER_STATS_OUTPUT_FILE="${_ru_stem}_run_${RUN_REPEAT_INDEX}${_ru_ext}"
	fi
}

whisper_build_training_args() {
	local DATA_PATH_LITERAL='${COMP597_JOB_STUDENT_STORAGE_DIR}/synthetic_whisper_data.pt'
	ARGS=(
		--logging.level INFO
		--model whisper
		--trainer simple
		--batch_size "${WHISPER_BATCH_SIZE}"
		--learning_rate "${WHISPER_LEARNING_RATE}"
		--data synthetic_whisper
		--data_configs.synthetic_whisper.data_path "${DATA_PATH_LITERAL}"
		--data_configs.synthetic_whisper.num_unique_samples "${WHISPER_NUM_UNIQUE_SAMPLES}"
		--data_configs.synthetic_whisper.num_workers "${WHISPER_NUM_WORKERS}"
		--data_configs.synthetic_whisper.repeat "${WHISPER_REPEAT}"
		--data_configs.synthetic_whisper.memory_only "${WHISPER_MEMORY_ONLY}"
		--data_configs.synthetic_whisper.data_type "${WHISPER_DATA_TYPE}"
		--data_configs.synthetic_whisper.chunk_size "${WHISPER_CHUNK_SIZE}"
		--data_configs.synthetic_whisper.num_shards "${WHISPER_NUM_SHARDS}"
		--data_configs.synthetic_whisper.force_regenerate "${WHISPER_FORCE_REGENERATE}"
		--trainer_stats "${WHISPER_TRAINER_STATS}"
	)
	case "${WHISPER_TRAINER_STATS}" in
		codecarbon)
			ARGS+=(
				--trainer_stats_configs.codecarbon.run_num "${WHISPER_CODECARBON_RUN_NUM}"
				--trainer_stats_configs.codecarbon.project_name whisper
				--trainer_stats_configs.codecarbon.output_dir "${WHISPER_OUTPUT_DIR}"
			)
			;;
		codecarbon_e2e)
			ARGS+=(
				--trainer_stats_configs.codecarbon_e2e.run_num "${WHISPER_CODECARBON_RUN_NUM}"
				--trainer_stats_configs.codecarbon_e2e.project_name whisper
				--trainer_stats_configs.codecarbon_e2e.output_dir "${WHISPER_OUTPUT_DIR}"
			)
			;;
		codecarbon_coarse)
			ARGS+=(
				--trainer_stats_configs.codecarbon_coarse.run_num "${WHISPER_CODECARBON_RUN_NUM}"
				--trainer_stats_configs.codecarbon_coarse.project_name whisper
				--trainer_stats_configs.codecarbon_coarse.output_dir "${WHISPER_OUTPUT_DIR}"
				--trainer_stats_configs.codecarbon_coarse.step_interval "${WHISPER_CODECARBON_COARSE_STEP_INTERVAL}"
			)
			;;
		resource_util)
			ARGS+=(
				--trainer_stats_configs.resource_util.output_dir "${WHISPER_OUTPUT_DIR}"
				--trainer_stats_configs.resource_util.output_file "${WHISPER_STATS_OUTPUT_FILE}"
			)
			;;
		phase_timing)
			ARGS+=(
				--trainer_stats_configs.phase_timing.run_num "${WHISPER_CODECARBON_RUN_NUM}"
				--trainer_stats_configs.phase_timing.output_dir "${WHISPER_OUTPUT_DIR}"
			)
			;;
	esac
}
