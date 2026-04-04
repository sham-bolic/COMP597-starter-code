#!/usr/bin/env bash
# Sequential sweep for one data_type only (safer walltime than monolithic).
#
# From repo root:
#   DT=shard WHISPER_REPEAT_COUNT=3 ./scripts/sweep_whisper_sequential_one_data_type.sh
# For DT=memory, repeat = WHISPER_MEMORY_TARGET_SAMPLES / batch_size (default 16000).
#
# Outputs go under results/data_<backend>/batch_*_worker_* (same names as compare_data_type_results).
# Override parent: WHISPER_SWEEP_PARENT=results_2 DT=shard ./scripts/sweep_whisper_sequential_one_data_type.sh
#
# DT must be one of: memory, shard, chunks, memmap

set -euo pipefail

ROOT="$(readlink -f "$(dirname "$0")/..")"
cd "$ROOT"

: "${WHISPER_REPEAT_COUNT:=3}"
: "${WHISPER_MEMORY_TARGET_SAMPLES:=16000}"
: "${WHISPER_SWEEP_PARENT:=results}"
dt="${DT:?Set DT to memory, shard, chunks, or memmap}"

case "$dt" in
	memory) workers=(0 2) ; sweep_subdir=data_memory ;;
	shard) workers=(0 2) ; sweep_subdir=data_shard ;;
	chunks) workers=(0 2) ; sweep_subdir=data_chunk ;;
	memmap) workers=(0) ; sweep_subdir=data_memmap ;;
	*) echo "$(basename "$0"): unknown DT=$dt (use memory, shard, chunks, memmap)" >&2; exit 1 ;;
esac
export WHISPER_SWEEP_DATA_ROOT="${WHISPER_SWEEP_PARENT}/${sweep_subdir}"
_cells_per_dt=$((3 * ${#workers[@]}))

echo "=====================================================================" >&2
echo "[$(basename "$0")] DT=${dt}  (output root: ${WHISPER_SWEEP_DATA_ROOT})" >&2
echo "[$(basename "$0")]   ${_cells_per_dt} batch×worker cell(s)" >&2
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

echo "$(basename "$0"): done DT=$dt" >&2
