#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/strip_cr.py"
chmod +x "${SCRIPT_DIR}"/*.sh "${SCRIPT_DIR}"/*.sbatch
grep -n nproc_per_node "${SCRIPT_DIR}/qwen3.5-4b-dflash-linear-1epoch.yaml"
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/train-1epoch
sbatch "${SCRIPT_DIR}/cluster-dflash-linear-1epoch.sbatch"
squeue -u naqin
echo DONE
