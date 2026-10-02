#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --job-name=pb-regen
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/perfectblend-qwen35-4b/logs/regen-%j.out
#SBATCH --requeue

set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} SHARD_ID=${SHARD_ID:-unset} dollar0=$0 pwd=$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FALLBACK="/shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/perfectblend-regen"
if [[ ! -f "${SCRIPT_DIR}/run-regen.sh" ]]; then
  echo "SCRIPT_DIR ${SCRIPT_DIR} missing run-regen.sh; using ${FALLBACK}"
  SCRIPT_DIR="${FALLBACK}"
fi
echo "SCRIPT_DIR=${SCRIPT_DIR}"
if [[ -z "${SHARD_ID:-}" ]]; then
  echo "FAIL: SHARD_ID is required"
  exit 1
fi
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/perfectblend-qwen35-4b/logs
cd "${SCRIPT_DIR}"
bash "${SCRIPT_DIR}/run-regen.sh" "${SHARD_ID}"
