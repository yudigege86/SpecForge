#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --job-name=dflin-pipe
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/mal-eval/pipeline-sbatch-%j.out
#SBATCH --requeue
#SBATCH --export=ALL

set -euo pipefail
ROOT="/shared_nfs/naqin/Linear-Context-DFlash"
bash "${ROOT}/mal-eval/overlap-pipeline.sh"
