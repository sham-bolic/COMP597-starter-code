#!/usr/bin/env bash
# Run a command N times. Each invocation sees RUN_REPEAT_INDEX=1..N in the environment.
#
# Usage:
#   ./scripts/run_repeat.sh 5 echo hello
#   ./scripts/run_repeat.sh -e WHISPER_CODECARBON_RUN_NUM 3 ./scripts/start-whisper.sh
#   ./scripts/run_repeat.sh -c 10 ./maybe_fails.sh   # keep going if a run fails
#
set -euo pipefail

SCRIPT_NAME=$(basename "$0")
CONTINUE_ON_ERROR=0
EXPORT_INDEX_VAR=""

usage() {
	cat <<EOF
Usage:
  $SCRIPT_NAME [-c] [-e VAR] <count> <command> [args...]

  <count>    positive integer
  <command>  command to run (use ./path or quote if it contains spaces)

Options:
  -e VAR     export VAR=<1..count> for each run (in addition to RUN_REPEAT_INDEX)
  -c         continue after a non-zero exit (default: stop on first failure)

Environment set for each run:
  RUN_REPEAT_INDEX   current run number (1-based)

Examples:
  $SCRIPT_NAME 4 python3 scripts/plot_all_resource_util.py results/data/test/resource_util.csv
  $SCRIPT_NAME -e WHISPER_CODECARBON_RUN_NUM 3 ./scripts/start-whisper.sh --trainer_stats codecarbon
EOF
}

while [[ $# -gt 0 && "${1:0:1}" == "-" ]]; do
	case "$1" in
		-c | --continue-on-error)
			CONTINUE_ON_ERROR=1
			shift
			;;
		-e | --export-equals)
			[[ $# -ge 2 ]] || {
				echo "$SCRIPT_NAME: -e requires a variable name" >&2
				exit 1
			}
			EXPORT_INDEX_VAR="$2"
			shift 2
			;;
		-h | --help)
			usage
			exit 0
			;;
		*)
			echo "$SCRIPT_NAME: unknown option: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

[[ $# -ge 2 ]] || {
	usage >&2
	exit 1
}

COUNT="$1"
shift

[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || {
	echo "$SCRIPT_NAME: count must be a positive integer, got: $COUNT" >&2
	exit 1
}

for ((i = 1; i <= COUNT; i++)); do
	echo "[$SCRIPT_NAME] run $i / $COUNT" >&2
	export RUN_REPEAT_INDEX="$i"
	if [[ -n "$EXPORT_INDEX_VAR" ]]; then
		export "$EXPORT_INDEX_VAR"="$i"
	fi
	set +e
	"$@"
	status=$?
	set -e
	if [[ "$status" -ne 0 ]]; then
		echo "[$SCRIPT_NAME] run $i exited with $status" >&2
		if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
			exit "$status"
		fi
	fi
done
