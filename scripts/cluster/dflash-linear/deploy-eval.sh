#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/strip_cr.py"
chmod +x "${SCRIPT_DIR}"/*.sh "${SCRIPT_DIR}"/*.sbatch
test -f specforge/modeling/draft/dflash_linear.py
test -f /shared_nfs/naqin/Linear-Context-DFlash/train-1epoch/20260918T192223Z/train/qwen3.5-4b-dflash-linear-1epoch-latest/training_state.pt
test -s /shared_nfs/naqin/primus-specforge-smoke/sharegpt_eval.holdout.jsonl
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/eval-1epoch
sbatch "${SCRIPT_DIR}/cluster-dflash-linear-eval.sbatch"
squeue -u naqin
echo DONE
