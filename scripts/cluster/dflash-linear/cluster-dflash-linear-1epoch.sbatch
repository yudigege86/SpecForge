#!/bin/bash
#SBATCH --account=amd-brain-models
# brain-models is at QOSGrpNodeLimit (~8 running); burst is preemptible.
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --job-name=dflin-1ep
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/train-1epoch/sbatch-%j.out
#SBATCH --requeue
# Do not pack onto the running sf-online allocation.
#SBATCH --exclude=crsuse2-m2m-012,crsuse2-m2m-023,crsuse2-m2m-042,crsuse2-m2m-150,crsuse2-m2m-226,crsuse2-m2m-251

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/train-1epoch
cd "${SCRIPT_DIR}"
bash run-dflash-linear-1epoch.sh
