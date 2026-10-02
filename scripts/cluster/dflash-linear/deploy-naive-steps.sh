#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/strip_cr.py"
chmod +x "${SCRIPT_DIR}"/*.sh "${SCRIPT_DIR}"/*.sbatch
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/naive-steps
sbatch "${SCRIPT_DIR}/cluster-dflash-linear-naive-steps.sbatch"
squeue -u naqin
echo DONE
