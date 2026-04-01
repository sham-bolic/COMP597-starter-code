#!/bin/bash
# Run on the Slurm compute node: full batch × worker grid inside ONE allocation (no nested srun).
# Invoked as: COMP597_JOB_COMMAND=bash ... job.sh bash sweep_whisper_grid_inner.sh --trainer_stats ... ...
set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")
cd "${REPO_DIR}"

_normalize_list_words() {
	tr ', ' '\n' | sed '/^[[:space:]]*$/d' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

trainer_stats=""
batch_spec=""
worker_spec=""
extras=()

while [[ $# -gt 0 ]]; do
	case "$1" in
		--trainer_stats)
			[[ $# -lt 2 ]] && {
				echo "$(basename "$0"): --trainer_stats requires a value" >&2
				exit 1
			}
			trainer_stats="$2"
			shift 2
			;;
		--batch_sizes)
			[[ $# -lt 2 ]] && {
				echo "$(basename "$0"): --batch_sizes requires a value" >&2
				exit 1
			}
			batch_spec="$2"
			shift 2
			;;
		--worker_sizes)
			[[ $# -lt 2 ]] && {
				echo "$(basename "$0"): --worker_sizes requires a value" >&2
				exit 1
			}
			worker_spec="$2"
			shift 2
			;;
		--)
			shift
			extras+=("$@")
			break
			;;
		*)
			extras+=("$1")
			shift
			;;
	esac
done

[[ -n "$trainer_stats" && -n "$batch_spec" && -n "$worker_spec" ]] || {
	echo "Usage: $(basename "$0") --trainer_stats NAME --batch_sizes LIST --worker_sizes LIST [-- EXTRA]" >&2
	exit 1
}

mapfile -t batches < <(echo "$batch_spec" | _normalize_list_words)
mapfile -t workers < <(echo "$worker_spec" | _normalize_list_words)
((${#batches[@]} > 0 && ${#workers[@]} > 0)) || {
	echo "$(basename "$0"): empty batch or worker list" >&2
	exit 1
}

total=$((${#batches[@]} * ${#workers[@]}))
_repeat="${WHISPER_REPEAT_COUNT:-3}"
idx=0
echo "Single-job grid (on-node): trainer_stats=${trainer_stats}, ${total} cell(s), ${_repeat} run(s) per cell (WHISPER_REPEAT_COUNT)" >&2

for b in "${batches[@]}"; do
	for w in "${workers[@]}"; do
		idx=$((idx + 1))
		echo "[${idx}/${total}] batch_size=${b} num_workers=${w}" >&2
		run_repeat_cmd=(
			"${SCRIPTS_DIR}/run_repeat.sh"
			-e
			WHISPER_CODECARBON_RUN_NUM
			"${_repeat}"
			bash
			"${SCRIPTS_DIR}/whisper_run_one_cell.sh"
			"${extras[@]}"
			--batch_size "$b"
			--num_workers "$w"
			--trainer_stats "$trainer_stats"
		)
		"${run_repeat_cmd[@]}"
	done
done
