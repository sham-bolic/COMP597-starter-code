#!/bin/bash
# Single entry point for Whisper + synthetic_whisper on Slurm.
# Learning rate defaults to 1e-6; override with --learning_rate or WHISPER_LEARNING_RATE.
# Synthetic pool size = num_unique_samples (default 16000); --batch_size is training mini-batch only.
# Optional: --repeat, --memory_only (RAM pool = --batch_size unique samples), --data_type, --chunk_size, --num_shards.
# Default trainer stats: codecarbon. For resource-util CSV logging:
#   ./start-whisper.sh --trainer_stats resource_util
# For CUDA-synced forward/backward/optimizer timing CSVs (results under WHISPER_OUTPUT_DIR):
#   ./start-whisper.sh --trainer_stats phase_timing
# Optional warmup for summary stats only: --trainer_stats_configs.phase_timing.warmup_steps N
# Time only one sub-phase per run (CUDA sync on that interval only): e.g.
#   --trainer_stats_configs.phase_timing.measure_phase forward
# (also: backward, optimizer, step, or all)
#
# Slurm driver (default blocking srun; use sbatch for async submit + log files):
#   WHISPER_SLURM_LAUNCHER=sbatch.sh ./scripts/start-whisper.sh ...
# Default is srun.sh. sbatch.sh writes logs per default_sbatch_config.sh (e.g. comp597-%N-%j.log).

_this="${BASH_SOURCE[0]:-$0}"
SCRIPTS_DIR=$(readlink -f -n "$(dirname "$_this")")
[[ -n "${SCRIPTS_DIR}" ]] || {
	echo "start-whisper.sh: could not resolve scripts directory from ${_this}" >&2
	exit 1
}
# shellcheck source=whisper_srun_hyperparams.sh
. "${SCRIPTS_DIR}/whisper_srun_hyperparams.sh"
# shellcheck source=whisper_launch_cell_body.sh
. "${SCRIPTS_DIR}/whisper_launch_cell_body.sh"

WHISPER_DEFAULT_TRAINER_STATS=codecarbon
whisper_parse_hyperparams "$@"
whisper_apply_run_repeat_resource_util_name

# So trainer_stats noop can write under the same results/data/batch_*_worker_* tree.
export WHISPER_OUTPUT_DIR

whisper_build_training_args

WHISPER_SLURM_LAUNCHER="${WHISPER_SLURM_LAUNCHER:-srun.sh}"
# Undo common env typo: WHISPER_SLURM_LAUNCHER==foo.sh assigns leading '=' (bash treats VAR==x as VAR='=x').
if [[ "${WHISPER_SLURM_LAUNCHER}" == =* ]]; then
	echo "start-whisper.sh: WHISPER_SLURM_LAUNCHER had a leading '=' (often from VAR==path). Using ${WHISPER_SLURM_LAUNCHER#=}." >&2
	WHISPER_SLURM_LAUNCHER="${WHISPER_SLURM_LAUNCHER#=}"
fi
if [[ "${WHISPER_SLURM_LAUNCHER}" == /* ]]; then
	_slurm_driver="${WHISPER_SLURM_LAUNCHER}"
else
	_slurm_driver="${SCRIPTS_DIR}/${WHISPER_SLURM_LAUNCHER##*/}"
fi
[[ -f "${_slurm_driver}" ]] || {
	_bn=$(basename "${BASH_SOURCE[0]:-$0}")
	echo "${_bn}: WHISPER_SLURM_LAUNCHER=${WHISPER_SLURM_LAUNCHER} -> not found: ${_slurm_driver}" >&2
	echo "${_bn}: (expected Slurm driver next to this repo under scripts/, e.g. scripts/srun.sh. Check path and git sync on this host.)" >&2
	exit 1
}
"${_slurm_driver}" "${ARGS[@]}" "${WHISPER_REST[@]}"
