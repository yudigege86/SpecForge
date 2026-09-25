#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/strip_cr.py"
chmod +x "${SCRIPT_DIR}"/*.sh "${SCRIPT_DIR}"/*.sbatch
test -f scripts/eval/dflash_linear_capture_replay.py
test -f "${SCRIPT_DIR}/inside-dflash-linear-replay.sh"
test -f /shared_nfs/naqin/Linear-Context-DFlash/eval-1epoch/20260918T213552Z/draft_hf/config.json
test -d /shared_nfs/naqin/primus-specforge-smoke/qwen-capture-40k/valid-links
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/capture-replay
sbatch "${SCRIPT_DIR}/cluster-dflash-linear-replay.sbatch"
squeue -u naqin
echo DONE
