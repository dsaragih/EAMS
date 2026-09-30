#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_emnn.sh"
HELPER_SCRIPT="$SCRIPT_DIR/emnn_run_helpers.sh"
SELECTED_GROUPS=()
EXTRA_OVERRIDES=()

usage() {
    cat <<'EOF'
Usage: ./train_emnn_local.sh [config_name | "config_a config_b" ...] [-- hydra_override ...]

Runs one or more predefined EMNN training bundles locally via ./run_emnn.sh.

Examples:
  ./train_emnn_local.sh "INTRA_ARGS emnn_intra sched_plateau aug_tcgn"
  ./train_emnn_local.sh "INTRA_ARGS emnn_intra sched_plateau aug_tcgn" -- model.lr=0.0005
  sbatch train_emnn.sh "LIVER_ARGS emnn_liver sched_plateau2 liver_features_stable liver_local_continuity aug_tcgn"

With no arguments, runs the documented IntrA EAMS comparison.
Each quoted argument before '--' is one run; names inside it combine bundles.
Unquoted separate arguments create separate runs.
Arguments after '--' apply to every run; later Hydra overrides win.
See docs/EAMS_EXPERIMENT_RUNS.md for groups and main commands.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_OVERRIDES=("$@")
            break
            ;;
        *)
            SELECTED_GROUPS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -x "$RUN_SCRIPT" ]]; then
    echo "Expected executable launcher at $RUN_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$HELPER_SCRIPT" ]]; then
    echo "Expected helper script at $HELPER_SCRIPT" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$HELPER_SCRIPT"

RUN_GROUPS=("${DEFAULT_RUN_GROUPS[@]}")
if [[ "${#SELECTED_GROUPS[@]}" -gt 0 ]]; then
    RUN_GROUPS=("${SELECTED_GROUPS[@]}")
fi

for group_spec in "${RUN_GROUPS[@]}"; do
    group_args=()
    resolve_group_args "$group_spec" group_args
    echo "Running local config group: $(format_group_label "$group_spec")"
    "$RUN_SCRIPT" "${group_args[@]}" "${EXTRA_OVERRIDES[@]}"
done