#!/bin/bash
# Submit one burst regen job per shard that is not done and not already queued.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

if [[ -f /etc/profile.d/spur.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/spur.sh
fi

MODE="${1:-all}"
mkdir -p "${LOG_ROOT}" "${REGEN_DIR}"

PAUSE_FILE="${PAUSE_PB_REGEN_SUBMIT:-/shared_nfs/naqin/Linear-Context-DFlash/PAUSE_PB_REGEN_SUBMIT}"
if [[ -f "${PAUSE_FILE}" ]]; then
  echo "submit paused; ${PAUSE_FILE} exists"
  exit 0
fi

if [[ -f "${PROMPTS_DIR}/num_shards" ]]; then
  NUM_SHARDS="$(tr -d '[:space:]' < "${PROMPTS_DIR}/num_shards")"
fi

job_name_for() {
  echo "pb-r$1"
}

queued_names() {
  local names=""
  names="$(squeue -u naqin -h -o '%j' 2>/dev/null || true)"
  if [[ -z "${names}" ]]; then
    names="$(squeue -u naqin 2>/dev/null | awk 'NR>1 {print $3}' || true)"
  fi
  printf '%s\n' "${names}"
}

is_queued() {
  local name="$1"
  printf '%s\n' "${QUEUED_NAMES}" | grep -Fxq "${name}"
}

QUEUED_NAMES="$(queued_names || true)"

submit_one() {
  local shard="$1"
  local name
  name="$(job_name_for "${shard}")"
  local done="${REGEN_DIR}/shard-${shard}.done"
  local input
  if [[ "${shard}" == "smoke" ]]; then
    input="${PROMPTS_DIR}/shard-smoke.jsonl"
  else
    input="${PROMPTS_DIR}/shard-${shard}.jsonl"
  fi
  if [[ ! -s "${input}" ]]; then
    echo "skip ${shard}: missing input ${input}"
    return 0
  fi
  if [[ -f "${done}" ]]; then
    return 0
  fi
  if is_queued "${name}"; then
    return 0
  fi
  local err
  err="$(mktemp "${LOG_ROOT}/sbatch.XXXXXX")"
  local jobid
  if ! jobid="$(sbatch --parsable \
    --job-name="${name}" \
    --export=ALL,SHARD_ID="${shard}" \
    "${SCRIPT_DIR}/cluster-regen.sbatch" 2>"${err}")"; then
    if grep -q "QOSMaxSubmitJobPerUserLimit" "${err}"; then
      echo "qos submit limit reached; remaining shards will wait"
      rm -f "${err}"
      return 2
    fi
    echo "submit ${shard} failed:"
    cat "${err}"
    rm -f "${err}"
    return 1
  fi
  rm -f "${err}"
  QUEUED_NAMES="${QUEUED_NAMES}"$'\n'"${name}"
  echo "submitted ${shard} job=${jobid} name=${name}"
}

case "${MODE}" in
  smoke)
    submit_one smoke
    ;;
  all)
    i=0
    while [[ "${i}" -lt "${NUM_SHARDS}" ]]; do
      shard="$(printf '%04d' "${i}")"
      rc=0
      submit_one "${shard}" || rc=$?
      if [[ "${rc}" -eq 2 ]]; then
        break
      elif [[ "${rc}" -ne 0 ]]; then
        exit "${rc}"
      fi
      i=$((i + 1))
    done
    ;;
  *)
    submit_one "${MODE}"
    ;;
esac

if [[ "${MODE}" == "all" && "${NO_WATCH:-0}" != "1" ]]; then
  mkdir -p "${LOG_ROOT}"
  nohup bash "${SCRIPT_DIR}/watch-and-merge.sh" \
    > "${LOG_ROOT}/watch.out" 2>&1 &
  echo "watch_pid=$! log=${LOG_ROOT}/watch.out"
fi
