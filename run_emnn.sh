#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
VENV_DIR="${VENV_DIR:-$REPO_DIR/mesh-venv}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_ENTRYPOINT="$REPO_DIR/train_lightning.py"
DRY_RUN=0

sanitize_path_component() {
    local value="$1"
    value="${value##*/}"
    value="${value//[^A-Za-z0-9._-]/_}"
    printf '%s' "$value"
}

build_write_dir_middle_override() {
    local arg
    local data_name=""
    local encoder_name=""

    for arg in "$@"; do
        case "$arg" in
            data=*)
                data_name="${arg#*=}"
                ;;
            model/encoder_config=*)
                encoder_name="${arg#*=}"
                ;;
        esac
    done

    if [[ -z "$data_name" ]]; then
        data_name="default_data"
    fi
    if [[ -z "$encoder_name" ]]; then
        encoder_name="default_encoder"
    fi

    printf 'experiments/%s/%s' \
        "$(sanitize_path_component "$data_name")" \
        "$(sanitize_path_component "$encoder_name")"
}

has_explicit_write_dir_middle_override() {
    local arg
    for arg in "$@"; do
        case "$arg" in
            paths.write_dir_middle=*)
                return 0
                ;;
        esac
    done

    return 1
}

DEFAULT_ARGS=(
    "trainer.max_epochs=100"
)

usage() {
    cat <<'EOF'
Usage: ./run_emnn.sh [hydra_override ...]

Runs train_lightning.py with the shared EMNN environment bootstrap and default overrides.

Examples:
    ./run_emnn.sh model=emnn_teeth data=iosseg_graph k_folds=1
    ./run_emnn.sh --dry-run model=emnn_intra model/encoder_config=vertex_emnn data=intra_graph

Environment overrides:
  VENV_DIR    Python virtual environment path
  ENV_FILE    Optional .env file path to source
  PYTHON_BIN  Python executable to run
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ ! -f "$TRAIN_ENTRYPOINT" ]]; then
    echo "Expected training entrypoint at $TRAIN_ENTRYPOINT" >&2
    exit 1
fi

cd "$REPO_DIR"

if [[ -d "$VENV_DIR" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

export HYDRA_FULL_ERROR=1

RUN_ARGS=("${DEFAULT_ARGS[@]}")
if ! has_explicit_write_dir_middle_override "$@"; then
    WRITE_DIR_MIDDLE_OVERRIDE="paths.write_dir_middle=$(build_write_dir_middle_override "${DEFAULT_ARGS[@]}" "$@")"
    RUN_ARGS+=("$WRITE_DIR_MIDDLE_OVERRIDE")
fi
RUN_ARGS+=("$@")

echo "Running $TRAIN_ENTRYPOINT with overrides: ${RUN_ARGS[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' "$PYTHON_BIN" "$TRAIN_ENTRYPOINT" "${RUN_ARGS[@]}"
    printf '\n'
    exit 0
fi

"$PYTHON_BIN" "$TRAIN_ENTRYPOINT" "${RUN_ARGS[@]}"