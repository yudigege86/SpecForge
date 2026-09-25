#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=dflin-mal
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/eval-1epoch/sbatch-%j.out
#SBATCH --requeue
# Do not pack onto running naqin GPU jobs (sf-ref-8 is 044/272); refresh from squeue.
#SBATCH --exclude=crsuse2-m2m-012,crsuse2-m2m-023,crsuse2-m2m-042,crsuse2-m2m-044,crsuse2-m2m-049,crsuse2-m2m-150,crsuse2-m2m-226,crsuse2-m2m-251,crsuse2-m2m-272,crsuse2-m2m-275

set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} dollar0=$0 pwd=$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FALLBACK="/shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/dflash-linear"
if [[ ! -f "${SCRIPT_DIR}/run-dflash-linear-mal.sh" ]]; then
  echo "SCRIPT_DIR ${SCRIPT_DIR} missing run script; using ${FALLBACK}"
  SCRIPT_DIR="${FALLBACK}"
fi
echo "SCRIPT_DIR=${SCRIPT_DIR}"
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/eval-1epoch
cd "${SCRIPT_DIR}"
bash "${SCRIPT_DIR}/run-dflash-linear-mal.sh"
