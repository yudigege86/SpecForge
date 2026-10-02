#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --job-name=dflin-m1
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/m1-gates/sbatch-%j.out
#SBATCH --requeue
#SBATCH --export=ALL

set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} pwd=$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FALLBACK="/shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/dflash-linear"
if [[ ! -f "${SCRIPT_DIR}/run-dflash-linear-m1-gates.sh" ]]; then
  SCRIPT_DIR="${FALLBACK}"
fi
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/mal-eval/m1-gates
cd "${SCRIPT_DIR}"
bash "${SCRIPT_DIR}/run-dflash-linear-m1-gates.sh"
