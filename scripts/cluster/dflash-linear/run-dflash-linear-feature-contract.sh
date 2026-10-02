#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RESULTS_DIR="${RESULTS_DIR:-/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/feature-contract}"
export INSIDE_SCRIPT="/workspace/SpecForge/scripts/cluster/dflash-linear/inside-dflash-linear-feature-contract.sh"
export CONTAINER_NAME="dflash-linear-contract"
export HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-/shared_nfs/naqin/primus-specforge-smoke/qwen-capture-40k/valid-links}"
export EVAL_N="${EVAL_N:-16}"
exec bash "${SCRIPT_DIR}/run-dflash-linear-docker.sh"
