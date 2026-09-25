#!/bin/bash
#SBATCH --account=amd-brain-models
#SBATCH --qos=amd-burst-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=fla-scan
#SBATCH --output=/shared_nfs/naqin/Linear-Context-DFlash/fla-scan/sbatch-%j.out
#SBATCH --requeue

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /shared_nfs/naqin/Linear-Context-DFlash/fla-scan
cd "${SCRIPT_DIR}"
bash run-fla-scan-test.sh
