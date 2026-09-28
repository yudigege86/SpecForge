#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=pb-prep
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/perfectblend-qwen35-4b/logs/prepare-%j.out
#SBATCH --requeue

set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} dollar0=$0 pwd=$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FALLBACK="/shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/perfectblend-regen"
if [[ ! -f "${SCRIPT_DIR}/run-prepare.sh" ]]; then
  echo "SCRIPT_DIR ${SCRIPT_DIR} missing run-prepare.sh; using ${FALLBACK}"
  SCRIPT_DIR="${FALLBACK}"
fi
echo "SCRIPT_DIR=${SCRIPT_DIR}"
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/perfectblend-qwen35-4b/logs
cd "${SCRIPT_DIR}"
bash "${SCRIPT_DIR}/run-prepare.sh"
