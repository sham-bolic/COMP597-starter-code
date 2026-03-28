#!/bin/bash
# Shared helpers for scripts/start-whisper.sh: parse hyperparameters and pass the
# rest through to srun.sh / launch.py.
#
# Before calling whisper_parse_hyperparams, optionally set:
#   WHISPER_DEFAULT_BATCH_SIZE, WHISPER_DEFAULT_LEARNING_RATE,
#   WHISPER_DEFAULT_NUM_WORKERS, WHISPER_DEFAULT_TRAINER_STATS,
#   WHISPER_DEFAULT_CODECARBON_RUN_NUM, WHISPER_DEFAULT_OUTPUT_DIR,
#   WHISPER_DEFAULT_STATS_OUTPUT_FILE (legacy: WHISPER_DEFAULT_RESOURCE_UTIL_OUTPUT_FILE)
#   WHISPER_DEFAULT_N_SAMPLES, WHISPER_DEFAULT_REPEAT, WHISPER_DEFAULT_MEMORY_ONLY
#
# Or set WHISPER_BATCH_SIZE / WHISPER_LEARNING_RATE / ... in the environment.


whisper_print_help() {
	local name
	name=$(basename "$0")
	cat <<EOF
Usage: $name [hyperparameters] [extra launch.py args ...]

Optional hyperparameters (also settable via WHISPER_* environment variables):
  --batch_size N
  --learning_rate X   (optional; default 1e-6)
  --num_workers N
       (synonym: --data_configs.synthetic_whisper.num_workers N)
  --n_samples N
       (synonym: --data_configs.synthetic_whisper.n_samples N; milabench-style pool size)
  --repeat N
       (synonym: --data_configs.synthetic_whisper.repeat N; virtual length = n_samples * repeat)
  --memory_only 0|1
       (synonym: --data_configs.synthetic_whisper.memory_only; 1 = RAM only, 0 = use data_path cache)
  --trainer_stats NAME        e.g. codecarbon, resource_util, no-op
  --trainer_stats_configs.codecarbon.run_num N
  --output_dir PATH
       Directory for trainer-stats output (default: results/data/batch_<batch>_worker_<num_workers>,
       e.g. results/data/batch_4_worker_0). Shared by all backends. Override with this flag or
       WHISPER_OUTPUT_DIR / WHISPER_DEFAULT_OUTPUT_DIR.
  --stats_output_file NAME
       Basename of the resource_util CSV inside --output_dir (default:
       resource_util.csv). Ignored when --trainer_stats is codecarbon (that
       backend writes multiple run_*cc_* CSVs; see src/trainer/stats/codecarbon.py).
  --resource_util_output_file NAME
  --trainer_stats_configs.resource_util.output_file NAME
       Aliases for --stats_output_file

Any further arguments are forwarded unchanged to srun.sh (and then launch.py).

Environment defaults (used when a flag is not passed):
  WHISPER_BATCH_SIZE          (default from script, usually 4)
  WHISPER_LEARNING_RATE       (default 1e-6)
  WHISPER_NUM_WORKERS         (default 0)
  WHISPER_N_SAMPLES           (default 500)
  WHISPER_REPEAT              (dataset virtual repeat; default 1. For multi-job repeats see WHISPER_REPEAT_COUNT + start-whisper-3x.sh)
  WHISPER_MEMORY_ONLY         (default 1)
  WHISPER_TRAINER_STATS       (default from script)
  WHISPER_CODECARBON_RUN_NUM  (default 1)
  WHISPER_OUTPUT_DIR             (default results/data/batch_<batch>_worker_<workers>, or WHISPER_DEFAULT_OUTPUT_DIR)
  WHISPER_STATS_OUTPUT_FILE      (default resource_util.csv; resource_util only).
       Legacy env: WHISPER_RESOURCE_UTIL_OUTPUT_FILE / WHISPER_DEFAULT_RESOURCE_UTIL_OUTPUT_FILE
EOF
}

whisper_apply_hyperparam_defaults() {
	: "${WHISPER_BATCH_SIZE:=${WHISPER_DEFAULT_BATCH_SIZE:-4}}"
	: "${WHISPER_LEARNING_RATE:=${WHISPER_DEFAULT_LEARNING_RATE:-1e-6}}"
	: "${WHISPER_NUM_WORKERS:=${WHISPER_DEFAULT_NUM_WORKERS:-0}}"
	: "${WHISPER_TRAINER_STATS:=${WHISPER_DEFAULT_TRAINER_STATS:-codecarbon}}"
	: "${WHISPER_CODECARBON_RUN_NUM:=${WHISPER_DEFAULT_CODECARBON_RUN_NUM:-1}}"
	: "${WHISPER_STATS_OUTPUT_FILE:=${WHISPER_DEFAULT_STATS_OUTPUT_FILE:-${WHISPER_RESOURCE_UTIL_OUTPUT_FILE:-${WHISPER_DEFAULT_RESOURCE_UTIL_OUTPUT_FILE:-resource_util.csv}}}}"
	: "${WHISPER_N_SAMPLES:=${WHISPER_DEFAULT_N_SAMPLES:-500}}"
	: "${WHISPER_REPEAT:=${WHISPER_DEFAULT_REPEAT:-1}}"
	: "${WHISPER_MEMORY_ONLY:=${WHISPER_DEFAULT_MEMORY_ONLY:-1}}"
}

# Input: "$@". Sets WHISPER_* and WHISPER_REST (array).
whisper_parse_hyperparams() {
	whisper_apply_hyperparam_defaults
	local -a rest=()
	while [[ $# -gt 0 ]]; do
		case "$1" in
			--batch_size)
				[[ $# -lt 2 ]] && {
					echo "$(basename "$0"): --batch_size requires a value" >&2
					exit 1
				}
				WHISPER_BATCH_SIZE="$2"
				shift 2
				;;
			--learning_rate)
				[[ $# -lt 2 ]] && {
					echo "$(basename "$0"): --learning_rate requires a value" >&2
					exit 1
				}
				WHISPER_LEARNING_RATE="$2"
				shift 2
				;;
			--num_workers | --data_configs.synthetic_whisper.num_workers)
				[[ $# -lt 2 ]] && {
					echo "$(basename "$0"): $1 requires a value" >&2
					exit 1
				}
				WHISPER_NUM_WORKERS="$2"
				shift 2
				;;
			--n_samples | --data_configs.synthetic_whisper.n_samples)
				[[ $# -lt 2 ]] && {
					echo "$(basename "$0"): $1 requires a value" >&2
					exit 1
				}
				WHISPER_N_SAMPLES="$2"
				shift 2
				;;
			--repeat | --data_configs.synthetic_whisper.repeat)
				[[ $# -lt 2 ]] && {
					echo "$(basename "$0"): $1 requires a value" >&2
					exit 1
				}
				WHISPER_REPEAT="$2"
				shift 2
				;;
			--memory_only | --data_configs.synthetic_whisper.memory_only)
				[[ $# -lt 2 ]] && {
					echo "$(basename "$0"): $1 requires a value" >&2
					exit 1
				}
				WHISPER_MEMORY_ONLY="$2"
				shift 2
				;;
			--trainer_stats)
				[[ $# -lt 2 ]] && {
					echo "$(basename "$0"): --trainer_stats requires a value" >&2
					exit 1
				}
				WHISPER_TRAINER_STATS="$2"
				shift 2
				;;
			--trainer_stats_configs.codecarbon.run_num)
				[[ $# -lt 2 ]] && {
					echo "$(basename "$0"): --trainer_stats_configs.codecarbon.run_num requires a value" >&2
					exit 1
				}
				WHISPER_CODECARBON_RUN_NUM="$2"
				shift 2
				;;
			--output_dir)
				[[ $# -lt 2 ]] && {
					echo "$(basename "$0"): --output_dir requires a value" >&2
					exit 1
				}
				WHISPER_OUTPUT_DIR="$2"
				shift 2
				;;
			--stats_output_file | --resource_util_output_file | --trainer_stats_configs.resource_util.output_file)
				[[ $# -lt 2 ]] && {
					echo "$(basename "$0"): $1 requires a value" >&2
					exit 1
				}
				WHISPER_STATS_OUTPUT_FILE="$2"
				shift 2
				;;
			-h | --help)
				whisper_print_help
				exit 0
				;;
			*)
				rest+=("$1")
				shift
				;;
		esac
	done
	: "${WHISPER_OUTPUT_DIR:=${WHISPER_DEFAULT_OUTPUT_DIR:-results/data/batch_${WHISPER_BATCH_SIZE}_worker_${WHISPER_NUM_WORKERS}}}"
	WHISPER_REST=("${rest[@]}")
}
