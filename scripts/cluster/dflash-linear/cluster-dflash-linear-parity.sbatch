#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=dflin-parity
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/parity/sbatch-%j.out
#SBATCH --requeue
#SBATCH --export=ALL

set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} pwd=$(pwd)"
ENV_FILE="${LINEAR_PARITY_ENV:-/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/parity/submit.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FALLBACK="/shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/dflash-linear"
if [[ ! -f "${SCRIPT_DIR}/run-dflash-linear-parity.sh" ]]; then
  SCRIPT_DIR="${FALLBACK}"
fi
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/mal-eval/parity
cd "${SCRIPT_DIR}"
bash "${SCRIPT_DIR}/run-dflash-linear-parity.sh"
