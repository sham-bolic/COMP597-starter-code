#!/bin/bash
# Grid sweep: batch_size × num_workers for one --trainer_stats value.
#
# Default: each cell runs start-whisper-3x.sh → 3 job-level Slurm runs per (batch, workers)
# (WHISPER_REPEAT_COUNT, default 3; CodeCarbon run nums + resource_util _run_N.csv). Override with
# SWEEP_WHISPER_LAUNCHER=... or WHISPER_REPEAT_COUNT=1 for a single run per cell.
#
# Multi-job mode (default): each repeat is start-whisper.sh → srun (or sbatch via WHISPER_SLURM_LAUNCHER).
#
#   --single-job: one Slurm allocation; the full grid runs on the compute node via
#   sweep_whisper_grid_inner.sh (nested launch.sh / python only, no nested srun).

set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
LAUNCHER=${SWEEP_WHISPER_LAUNCHER:-"${SCRIPTS_DIR}/start-whisper-3x.sh"}

sweep_usage() {
	local n
	n=$(basename "$0")
	cat <<EOF
Usage: $n --trainer_stats NAME --batch_sizes LIST --worker_sizes LIST [options] [-- EXTRA]

Required:
  --trainer_stats NAME     Passed to each cell (resource_util, codecarbon, …)
  --batch_sizes LIST       Comma- or space-separated batch sizes
  --worker_sizes LIST      Comma- or space-separated DataLoader num_workers values

Optional:
  --single-job             One srun: entire grid runs inside that job (see scripts/sweep_whisper_grid_inner.sh).
                          sets COMP597_JOB_COMMAND=bash for that submit only (see config/default_job_config.sh).
  --dry-run                Print what would run
  -h, --help               Show this help

After --, remaining arguments are forwarded to each cell (shared hyperparameters).

Default per-cell output dirs: results/data/batch_<batch>_worker_<num_workers>/
Default: 3 training runs per cell (job-level repeats). Change with WHISPER_REPEAT_COUNT (e.g. 1 for one run).

Environment:
  SWEEP_WHISPER_LAUNCHER   Multi-job mode: launcher per cell (default: start-whisper-3x.sh)
  WHISPER_REPEAT_COUNT     Repeats per cell (default 3; see scripts/start-whisper-3x.sh)
EOF
}

_normalize_list_words() {
	tr ', ' '\n' | sed '/^[[:space:]]*$/d' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

trainer_stats=""
batch_spec=""
worker_spec=""
dry_run=0
single_job=0
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
		--single-job)
			single_job=1
			shift
			;;
		--dry-run)
			dry_run=1
			shift
			;;
		-h | --help)
			sweep_usage
			exit 0
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

[[ -n "$trainer_stats" ]] || {
	echo "$(basename "$0"): --trainer_stats is required" >&2
	sweep_usage >&2
	exit 1
}
[[ -n "$batch_spec" ]] || {
	echo "$(basename "$0"): --batch_sizes is required" >&2
	sweep_usage >&2
	exit 1
}
[[ -n "$worker_spec" ]] || {
	echo "$(basename "$0"): --worker_sizes is required" >&2
	sweep_usage >&2
	exit 1
}

mapfile -t batches < <(echo "$batch_spec" | _normalize_list_words)
mapfile -t workers < <(echo "$worker_spec" | _normalize_list_words)

((${#batches[@]} > 0)) || {
	echo "$(basename "$0"): --batch_sizes produced an empty list" >&2
	exit 1
}
((${#workers[@]} > 0)) || {
	echo "$(basename "$0"): --worker_sizes produced an empty list" >&2
	exit 1
}

run_launcher() {
	if [[ -x "$1" ]]; then
		"$@"
	else
		bash "$@"
	fi
}

if [[ "$single_job" -eq 1 ]]; then
	[[ -f "${SCRIPTS_DIR}/srun.sh" && -f "${SCRIPTS_DIR}/job.sh" && -f "${SCRIPTS_DIR}/sweep_whisper_grid_inner.sh" && -f "${SCRIPTS_DIR}/whisper_run_one_cell.sh" ]] || {
		echo "$(basename "$0"): missing script(s) under ${SCRIPTS_DIR} for --single-job" >&2
		exit 1
	}
	total=$((${#batches[@]} * ${#workers[@]}))
	echo "Single-job sweep: ${total} cell(s) inside one Slurm step (COMP597_JOB_COMMAND=bash → sweep_whisper_grid_inner.sh)" >&2
	if [[ "$dry_run" -eq 1 ]]; then
		printf '%q ' env COMP597_JOB_COMMAND="bash" "${SCRIPTS_DIR}/srun.sh" \
			"${SCRIPTS_DIR}/sweep_whisper_grid_inner.sh" \
			--trainer_stats "${trainer_stats}" \
			--batch_sizes "${batch_spec}" \
			--worker_sizes "${worker_spec}" \
			-- "${extras[@]}"
		echo
		exit 0
	fi
	run_launcher env COMP597_JOB_COMMAND="bash" "${SCRIPTS_DIR}/srun.sh" \
		"${SCRIPTS_DIR}/sweep_whisper_grid_inner.sh" \
		--trainer_stats "${trainer_stats}" \
		--batch_sizes "${batch_spec}" \
		--worker_sizes "${worker_spec}" \
		-- "${extras[@]}"
	exit 0
fi

[[ -f "$LAUNCHER" ]] || {
	echo "$(basename "$0"): launcher not found: $LAUNCHER" >&2
	exit 1
}

total=$((${#batches[@]} * ${#workers[@]}))
_repeat="${WHISPER_REPEAT_COUNT:-3}"
idx=0
echo "Sweep: trainer_stats=$trainer_stats, ${#batches[@]} × ${#workers[@]} = $total cell(s), ${_repeat} run(s) per cell (WHISPER_REPEAT_COUNT)" >&2
echo "Launcher: $LAUNCHER" >&2

for b in "${batches[@]}"; do
	for w in "${workers[@]}"; do
		idx=$((idx + 1))
		echo "[$idx/$total] batch_size=$b num_workers=$w" >&2
		cmd=(
			"$LAUNCHER"
			"${extras[@]}"
			--batch_size "$b"
			--num_workers "$w"
			--trainer_stats "$trainer_stats"
		)
		if [[ "$dry_run" -eq 1 ]]; then
			printf '%q ' "${cmd[@]}"
			echo
		else
			run_launcher "${cmd[@]}"
		fi
	done
done
