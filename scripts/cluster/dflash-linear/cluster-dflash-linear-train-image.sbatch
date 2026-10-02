#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=dflin-timg
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/docker-build/train-image-%j.out
#SBATCH --requeue
#SBATCH --export=ALL

set -euo pipefail
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/docker-build
SCRIPT="/shared_nfs/naqin/Linear-Context-DFlash/SpecForge/scripts/cluster/dflash-linear/build-sglang-0.5.18-train-image.sh"
bash "${SCRIPT}"
