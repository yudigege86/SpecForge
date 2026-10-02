#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RESULTS_DIR="${RESULTS_DIR:-/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/${EVAL_DATASET:-humaneval}-linear-parity}"
export INSIDE_SCRIPT="/workspace/SpecForge/scripts/cluster/dflash-linear/inside-dflash-linear-parity.sh"
export CONTAINER_NAME="dflash-linear-parity"
export SGLANG_DFLASH_LINEAR_SHADOW_CHECK="${SGLANG_DFLASH_LINEAR_SHADOW_CHECK:-1}"
exec bash "${SCRIPT_DIR}/run-dflash-linear-docker.sh"
