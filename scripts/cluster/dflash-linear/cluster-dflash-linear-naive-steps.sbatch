#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=dflin-naive
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/naive-steps/sbatch-%j.out
#SBATCH --requeue
# Do not pack onto the running sf-online allocation.
#SBATCH --exclude=crsuse2-m2m-023,crsuse2-m2m-042,crsuse2-m2m-150,crsuse2-m2m-226,crsuse2-m2m-251

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/naive-steps
cd "${SCRIPT_DIR}"
bash run-dflash-linear-naive-steps.sh
