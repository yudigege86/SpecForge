#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=dflin-sbsgl
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/speedbench-sglang/sbatch-%j.out
#SBATCH --requeue
#SBATCH --export=ALL
# Do not pack onto running naqin GPU jobs; refresh from squeue before submit.
#SBATCH --exclude=crsuse2-m2m-012,crsuse2-m2m-023,crsuse2-m2m-042,crsuse2-m2m-044,crsuse2-m2m-049,crsuse2-m2m-057,crsuse2-m2m-150,crsuse2-m2m-169,crsuse2-m2m-213,crsuse2-m2m-226,crsuse2-m2m-245,crsuse2-m2m-251,crsuse2-m2m-268,crsuse2-m2m-272,crsuse2-m2m-275

set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} dollar0=$0 pwd=$(pwd)"
ENV_FILE="${SPEEDBENCH_SGLANG_ENV:-/shared_nfs/naqin/Linear-Context-DFlash/speedbench-sglang/submit.env}"
if [[ -f "${ENV_FILE}" ]]; then
  echo "sourcing ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FALLBACK="/shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/dflash-linear"
if [[ ! -f "${SCRIPT_DIR}/run-dflash-linear-speedbench-sglang.sh" ]]; then
  echo "SCRIPT_DIR ${SCRIPT_DIR} missing run script; using ${FALLBACK}"
  SCRIPT_DIR="${FALLBACK}"
fi
echo "SCRIPT_DIR=${SCRIPT_DIR}"
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/speedbench-sglang
cd "${SCRIPT_DIR}"
bash "${SCRIPT_DIR}/run-dflash-linear-speedbench-sglang.sh"
