#!/bin/bash
#SBATCH --account=amd-brain-models
# 1-epoch DFlash2-linear train on burst (preemptible; --requeue). Partition wall is 1 day.
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --job-name=df2lin-1ep
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/train-dflash2-linear-1epoch/sbatch-%j.out
#SBATCH --requeue
#SBATCH --export=ALL

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FALLBACK="/shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/dflash-linear"
if [[ ! -f "${SCRIPT_DIR}/run-dflash2-linear-1epoch.sh" ]]; then
  SCRIPT_DIR="${FALLBACK}"
fi
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/train-dflash2-linear-1epoch
cd "${SCRIPT_DIR}"
bash "${SCRIPT_DIR}/run-dflash2-linear-1epoch.sh"
