#!/usr/bin/env bash
# Full sequential sweep: data_type outer loop, generate on Slurm once per disk type,
# then start-whisper-3x for each batch/worker/trainer_stats configuration.
#
# From repo root:
#   WHISPER_REPEAT_COUNT=3 ./scripts/sweep_whisper_sequential_monolithic.sh
# For data_type=memory, repeat = WHISPER_MEMORY_TARGET_SAMPLES / batch_size (default 16000).
#
# Each data_type writes under a separate tree (default parent dir results/):
#   results/data_memory/batch_*_worker_*  results/data_shard/...  results/data_chunk/...  results/data_memmap/...
# Override parent: WHISPER_SWEEP_PARENT=results_2 ./scripts/sweep_whisper_sequential_monolithic.sh
#
# Requires: scripts/generate_synthetic_whisper_data_srun.sh, scripts/start-whisper-3x.sh

set -euo pipefail

ROOT="$(readlink -f "$(dirname "$0")/..")"
cd "$ROOT"

: "${WHISPER_REPEAT_COUNT:=3}"
# For data_type=memory, len = batch_size * repeat; match ~num_unique_samples disk runs.
: "${WHISPER_MEMORY_TARGET_SAMPLES:=16000}"
: "${WHISPER_SWEEP_PARENT:=results}"

_dt_i=0
for dt in shard chunks memmap; do
	_dt_i=$((_dt_i + 1))
	case "$dt" in
		memory) workers=(0 2) ; sweep_subdir=data_memory ;;
		shard) workers=(0 2) ; sweep_subdir=data_shard ;;
		chunks) workers=(0 2) ; sweep_subdir=data_chunk ;;
		memmap) workers=(0) ; sweep_subdir=data_memmap ;;
	esac
	export WHISPER_SWEEP_DATA_ROOT="${WHISPER_SWEEP_PARENT}/${sweep_subdir}"
	_cells_per_dt=$((3 * ${#workers[@]}))

	echo "=====================================================================" >&2
	echo "[$(basename "$0")] data_type ${_dt_i}/4: ${dt}  (output root: ${WHISPER_SWEEP_DATA_ROOT})" >&2
	echo "[$(basename "$0")]   ${_cells_per_dt} batch×worker cell(s) this data_type" >&2
	echo "=====================================================================" >&2

	if [[ "$dt" != memory ]]; then
		gen_args=(--data_type "$dt")
		if [[ "$dt" == shard ]]; then
			gen_args+=(--num_shards 40)
		fi
		echo "[$(basename "$0")] (${dt}) running Slurm generate: ${gen_args[*]}" >&2
		./scripts/generate_synthetic_whisper_data_srun.sh "${gen_args[@]}"
	fi

	_cell_i=0
	for bs in 32 64 128; do
		for nw in "${workers[@]}"; do
			_cell_i=$((_cell_i + 1))
			echo "---------------------------------------------------------------------" >&2
			echo "[$(basename "$0")] (${dt}) cell ${_cell_i}/${_cells_per_dt}  batch_size=${bs}  num_workers=${nw}" >&2
			echo "---------------------------------------------------------------------" >&2

			common=(
				--data_type "$dt"
				--batch_size "$bs"
				--num_workers "$nw"
			)

			if [[ "$dt" == shard ]]; then
				common+=(--num_shards 40)
			fi
			if [[ "$dt" == memory ]]; then
				common+=(--repeat $((WHISPER_MEMORY_TARGET_SAMPLES / bs)))
			fi

			./scripts/start-whisper-3x.sh "${common[@]}" --trainer_stats codecarbon
			./scripts/start-whisper-3x.sh "${common[@]}" --trainer_stats resource_util
			./scripts/start-whisper-3x.sh "${common[@]}" --trainer_stats noop
			./scripts/start-whisper-3x.sh "${common[@]}" --trainer_stats codecarbon_e2e

			for ph in forward backward optimizer; do
				./scripts/start-whisper-3x.sh \
					"${common[@]}" \
					--trainer_stats phase_timing \
					--trainer_stats_configs.phase_timing.measure_phase "$ph"
			done
		done
	done
done

echo "$(basename "$0"): done." >&2
