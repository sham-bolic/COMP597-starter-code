#!/usr/bin/env bash
# Run phase_timing once per sub-phase (forward, backward, optimizer), each with
# job-level repeats via start-whisper-3x.sh (default 3 runs → run_1, run_2, run_3).
#
# Usage (from repo root):
#   ./scripts/run_phase_timing_each_phase_3x.sh
#   ./scripts/run_phase_timing_each_phase_3x.sh --batch_size 16 --num_workers 2
#   WHISPER_REPEAT_COUNT=5 ./scripts/run_phase_timing_each_phase_3x.sh --batch_size 8
#   WHISPER_SWEEP_DATA_ROOT=results/data_phase_timing ./scripts/run_phase_timing_each_phase_3x.sh
#
# All arguments are forwarded to start-whisper-3x.sh after the fixed --trainer_stats flags.
# Outputs use distinct basenames: ..._phase_timing_measure_forward_..., ..._backward_..., etc.

set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
LAUNCHER="${SCRIPTS_DIR}/start-whisper-3x.sh"
REPEAT="${WHISPER_REPEAT_COUNT:-3}"

# Optional: space-separated list, e.g. PHASE_TIMING_PHASES="forward backward optimizer step"
if [[ -n "${PHASE_TIMING_PHASES:-}" ]]; then
	read -r -a phases <<<"${PHASE_TIMING_PHASES}"
else
	phases=(forward backward optimizer)
fi

usage() {
	cat <<EOF
Usage: $(basename "$0") [args for start-whisper-3x.sh ...]

Runs phase_timing with measure_phase set to each of: ${phases[*]}
Each phase uses WHISPER_REPEAT_COUNT job repeats (default 3).

Environment:
  WHISPER_REPEAT_COUNT     Repeats per phase (default 3)
  PHASE_TIMING_PHASES      Override phases (space-separated)
  WHISPER_SWEEP_DATA_ROOT  Optional parent for results/.../batch_*_worker_*

Examples:
  $0 --batch_size 4 --num_workers 0
  WHISPER_REPEAT_COUNT=1 $0 --batch_size 16 --num_workers 2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

[[ -f "$LAUNCHER" ]] || {
	echo "$(basename "$0"): missing $LAUNCHER" >&2
	exit 1
}

for ph in "${phases[@]}"; do
	echo "=====================================================================" >&2
	echo "phase_timing  measure_phase=$ph  WHISPER_REPEAT_COUNT=$REPEAT" >&2
	echo "=====================================================================" >&2
	WHISPER_REPEAT_COUNT="$REPEAT" "$LAUNCHER" \
		--trainer_stats phase_timing \
		--trainer_stats_configs.phase_timing.measure_phase "$ph" \
		"$@"
done

echo "$(basename "$0"): done; phases=${phases[*]}" >&2
