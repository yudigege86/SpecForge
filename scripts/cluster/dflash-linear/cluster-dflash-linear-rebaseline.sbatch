#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --job-name=dflin-rebase
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/rebaseline-0.5.18/sbatch-%j.out
#SBATCH --requeue
#SBATCH --export=ALL

set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FALLBACK="/shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/dflash-linear"
if [[ ! -f "${SCRIPT_DIR}/run-dflash-linear-speedbench-sglang.sh" ]]; then
  SCRIPT_DIR="${FALLBACK}"
fi
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/mal-eval/rebaseline-0.5.18
export RUNTIME_IMAGE="${RUNTIME_IMAGE:-naqin/primus-specforge:v0.5.18-dflash-linear-rocm700-mi35x}"
export IMAGE_ARCHIVE="${IMAGE_ARCHIVE:-/shared_nfs/naqin/docker-images/primus-specforge-v0.5.18-dflash-linear-rocm700-mi35x.tar.zst}"
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
export DRAFT_HF="${DRAFT_HF:-z-lab/Qwen3.5-4B-DFlash}"
export EVAL_DATASET=humaneval
export ENABLE_THINKING=1
export MAX_NEW_TOKENS=4096
export REPLAY_OFFLINE=1
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"

echo "=== stock HE n=16 ==="
export EVAL_N=16
export RESULTS_DIR="/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/rebaseline-0.5.18/he16"
bash "${SCRIPT_DIR}/run-dflash-linear-speedbench-sglang.sh"

echo "=== stock HE n=164 ==="
export EVAL_N=164
export RESULTS_DIR="/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/rebaseline-0.5.18/he164"
bash "${SCRIPT_DIR}/run-dflash-linear-speedbench-sglang.sh"
