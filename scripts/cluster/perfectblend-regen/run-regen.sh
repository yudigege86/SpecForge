#!/bin/bash
# One-node 8-GPU Open-PerfectBlend regeneration worker. Resumes by sample id.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

SHARD_ID="${1:-${SHARD_ID:-}}"
if [[ -z "${SHARD_ID}" ]]; then
  echo "FAIL: SHARD_ID is required"
  exit 1
fi

mkdir -p "${ROOT}" "${REGEN_DIR}" "${LOG_ROOT}" "${HF_HOME}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${LOG_ROOT}/shard-${SHARD_ID}/${SLURM_JOB_ID:-${STAMP}}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/run.log") 2>&1

echo "host=$(hostname) stamp=${STAMP} image=${RUNTIME_IMAGE}"
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-$(hostname)}"
echo "SHARD_ID=${SHARD_ID}"
echo "SPECFORGE_SRC=${SPECFORGE_SRC}"
echo "CONCURRENCY=${CONCURRENCY} MAX_LENGTH=${MAX_LENGTH}"

test -f "${SPECFORGE_SRC}/scripts/regenerate_train_data.py"
test -s "${IMAGE_ARCHIVE}"
if [[ "${SHARD_ID}" == "smoke" ]]; then
  INPUT_FILE="${PROMPTS_DIR}/shard-smoke.jsonl"
else
  INPUT_FILE="${PROMPTS_DIR}/shard-${SHARD_ID}.jsonl"
fi
test -s "${INPUT_FILE}"
OUTPUT_FILE="${REGEN_DIR}/shard-${SHARD_ID}.jsonl"
DONE_FILE="${REGEN_DIR}/shard-${SHARD_ID}.done"

run_docker() {
  if [ "${USE_SG_DOCKER:-0}" = 1 ]; then
    sg docker -c "$(printf '%q ' docker "$@")"
  else
    docker "$@"
  fi
}

if docker info >/dev/null 2>&1; then
  USE_SG_DOCKER=0
else
  echo "docker info failed on $(hostname); retrying under sg docker"
  USE_SG_DOCKER=1
  if ! run_docker info >/dev/null 2>&1; then
    echo "FAIL: docker unavailable on $(hostname)"
    exit 1
  fi
fi

if ! run_docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
  echo "loading ${RUNTIME_IMAGE} from ${IMAGE_ARCHIVE}"
  zstd -dc "${IMAGE_ARCHIVE}" | run_docker load
fi
run_docker image inspect "${RUNTIME_IMAGE}" --format '{{.Id}}'

NAME="pb-regen-${SHARD_ID}-${SLURM_JOB_ID:-${STAMP}}"
run_docker rm -f "${NAME}" >/dev/null 2>&1 || true

set +e
run_docker run --rm --name "${NAME}" \
  --network host --ipc=host --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  -e HF_HOME="${HF_HOME}" \
  -e HF_HUB_CACHE="${HF_HUB_CACHE}" \
  -e HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e SHARD_ID="${SHARD_ID}" \
  -e INPUT_FILE="${INPUT_FILE}" \
  -e OUTPUT_FILE="${OUTPUT_FILE}" \
  -e DONE_FILE="${DONE_FILE}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e CONCURRENCY="${CONCURRENCY}" \
  -e MAX_LENGTH="${MAX_LENGTH}" \
  -e CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
  -e RUN_DIR="${RUN_DIR}" \
  -v /shared_nfs:/shared_nfs \
  -v "${SPECFORGE_SRC}:/workspace/SpecForge" \
  -v "${SCRIPT_DIR}:/opt/perfectblend-regen:ro" \
  "${RUNTIME_IMAGE}" \
  bash /opt/perfectblend-regen/inside-regen.sh
rc=$?
set -e

echo "docker_exit=${rc}"
if [[ ${rc} -ne 0 ]]; then
  echo "FAIL: perfectblend regen shard ${SHARD_ID} exited ${rc}"
  exit "${rc}"
fi

echo "PASS: perfectblend regen shard ${SHARD_ID}"
echo "output=${OUTPUT_FILE}"
echo "run_dir=${RUN_DIR}"
