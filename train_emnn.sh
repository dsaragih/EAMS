#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=eams
#SBATCH --gpus-per-node=1
#SBATCH --output=eams-%j.out
set -euo pipefail
# Generic SLURM example: adjust resources and load site-specific modules before submission.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/run_emnn.sh" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
fi

RUN_SCRIPT="$REPO_DIR/run_emnn.sh"
HELPER_SCRIPT="$REPO_DIR/emnn_run_helpers.sh"
SELECTED_GROUPS=()
EXTRA_OVERRIDES=()

usage() {
    cat <<'EOF'
Usage: ./train_emnn.sh [config_name | "config_a config_b" ...] [-- hydra_override ...]

Runs one or more predefined EMNN training bundles via ./run_emnn.sh.

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

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
fi

if command -v srun >/dev/null 2>&1; then
    srun nvidia-smi
fi
ulimit -n 4096

RUN_GROUPS=("${DEFAULT_RUN_GROUPS[@]}")

if [[ "${#SELECTED_GROUPS[@]}" -gt 0 ]]; then
    RUN_GROUPS=("${SELECTED_GROUPS[@]}")
fi

echo "Selected config groups:"
echo "  ${RUN_GROUPS[@]}"

for group_spec in "${RUN_GROUPS[@]}"; do
    group_args=()
    resolve_group_args "$group_spec" group_args
    echo "Running config group: $(format_group_label "$group_spec")"
    "$RUN_SCRIPT" "${group_args[@]}" "${EXTRA_OVERRIDES[@]}"
done
