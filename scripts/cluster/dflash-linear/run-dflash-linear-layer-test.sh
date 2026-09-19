#!/bin/bash
# Run context-first GDN layer unit tests on one MI355X (naive backend; no FLA).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPECFORGE_SRC="${SPECFORGE_SRC:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SMOKE_DIR="${SMOKE_DIR:-${SCRIPT_DIR}}"

RUNTIME_IMAGE="${RUNTIME_IMAGE:-naqin/primus-specforge:v0.5.14-rocm700-mi35x}"
IMAGE_ARCHIVE="${IMAGE_ARCHIVE:-/shared_nfs/naqin/docker-images/primus-specforge-v0.5.14-rocm700-mi35x.tar.zst}"
RESULTS_DIR="${RESULTS_DIR:-/shared_nfs/naqin/Linear-Context-DFlash/layer-test}"
if [[ -z "${HIP_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  HIP_VISIBLE_DEVICES=0
  CUDA_VISIBLE_DEVICES=0
fi
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-0}}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}"

mkdir -p "${RESULTS_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULTS_DIR}/${STAMP}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/run.log") 2>&1

echo "host=$(hostname) stamp=${STAMP} image=${RUNTIME_IMAGE}"
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-$(hostname)}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SPECFORGE_SRC=${SPECFORGE_SRC}"

test -f "${SPECFORGE_SRC}/specforge/modeling/draft/dflash_linear.py"
test -f "${SPECFORGE_SRC}/specforge/algorithms/dflash_linear/model.py"
test -f "${SPECFORGE_SRC}/tests/test_modeling/test_dflash_linear_layer.py"

if ! docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
  echo "loading ${RUNTIME_IMAGE} from ${IMAGE_ARCHIVE}"
  test -s "${IMAGE_ARCHIVE}"
  zstd -dc "${IMAGE_ARCHIVE}" | docker load
fi
docker image inspect "${RUNTIME_IMAGE}" --format '{{.Id}}'

NAME="dflin-layer-${STAMP}"
docker rm -f "${NAME}" >/dev/null 2>&1 || true

set +e
docker run --rm --name "${NAME}" \
  --network host --ipc=host --shm-size=16g \
  --ulimit memlock=-1:-1 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -e RUN_DIR="${RUN_DIR}" \
  -v /shared_nfs:/shared_nfs \
  -v "${SPECFORGE_SRC}:/workspace/SpecForge" \
  "${RUNTIME_IMAGE}" \
  bash /workspace/SpecForge/scripts/cluster/dflash-linear/inside-dflash-linear-layer-test.sh
rc=$?
set -e

echo "docker_exit=${rc}"
if [[ ${rc} -ne 0 ]]; then
  echo "FAIL: linear-context layer tests exited ${rc}"
  exit "${rc}"
fi

echo "PASS: linear-context layer tests"
echo "run_dir=${RUN_DIR}"
