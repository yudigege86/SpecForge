#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=dflin-smoke
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/sglang-smoke/sbatch-%j.out
#SBATCH --requeue
#SBATCH --export=ALL

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FALLBACK="/shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/dflash-linear"
if [[ ! -f "${SCRIPT_DIR}/run-dflash-linear-sglang-smoke.sh" ]]; then
  SCRIPT_DIR="${FALLBACK}"
fi
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/mal-eval/sglang-smoke
cd "${SCRIPT_DIR}"
bash "${SCRIPT_DIR}/run-dflash-linear-sglang-smoke.sh"
