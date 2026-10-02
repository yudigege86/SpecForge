#!/bin/bash
# Release held burst jobs, resubmit missing shards, merge when every shard is done.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

if [[ -f /etc/profile.d/spur.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/spur.sh
fi

if [[ -f "${PROMPTS_DIR}/num_shards" ]]; then
  NUM_SHARDS="$(tr -d '[:space:]' < "${PROMPTS_DIR}/num_shards")"
fi

mkdir -p "${LOG_ROOT}"

release_holds() {
  local jobid name state reason extra
  local rows=""
  rows="$(squeue -u naqin -h -o '%A %j %T %r' 2>/dev/null || true)"
  if [[ -z "${rows}" ]]; then
    rows="$(squeue -u naqin 2>/dev/null | awk 'NR>1 {print $1, $3, $5, $0}' || true)"
  fi
  [[ -n "${rows}" ]] || return 0
  while read -r jobid name state reason extra; do
    [[ -n "${jobid:-}" ]] || continue
    case "${name}" in
      pb-r*|pb-regen|pb-prep) ;;
      *) continue ;;
    esac
    if [[ "${reason}" == *JobHoldMaxRequeue* || "${reason}" == *Hold* ]]; then
      echo "release ${jobid} name=${name} reason=${reason}"
      scontrol release "${jobid}" || true
    fi
  done <<< "${rows}" || true
  return 0
}

cancel_finished_jobs() {
  # A completed shard can still occupy a QOS slot while docker/sglang
  # tears down. Free those slots so the next shard can start.
  local jobid name state rest shard
  local rows=""
  rows="$(squeue -u naqin -h -o '%A %j %T' 2>/dev/null || true)"
  if [[ -z "${rows}" ]]; then
    rows="$(squeue -u naqin 2>/dev/null | awk 'NR>1 {print $1, $3, $5}' || true)"
  fi
  [[ -n "${rows}" ]] || return 0
  while read -r jobid name state rest; do
    [[ -n "${jobid:-}" ]] || continue
    case "${name}" in
      pb-r[0-9][0-9][0-9][0-9])
        shard="${name#pb-r}"
        ;;
      *) continue ;;
    esac
    if [[ -f "${REGEN_DIR}/shard-${shard}.done" ]]; then
      echo "scancel ${jobid} name=${name} shard already done"
      scancel "${jobid}" || true
    fi
  done <<< "${rows}" || true
  return 0
}

count_done() {
  local i=0
  local n=0
  while [[ "${i}" -lt "${NUM_SHARDS}" ]]; do
    shard="$(printf '%04d' "${i}")"
    if [[ -f "${REGEN_DIR}/shard-${shard}.done" ]]; then
      n=$((n + 1))
    fi
    i=$((i + 1))
  done
  echo "${n}"
}

while true; do
  date -u +%Y-%m-%dT%H:%M:%SZ > "${LOG_ROOT}/watch.heartbeat"
  release_holds || true
  cancel_finished_jobs || true
  done_n="$(count_done)"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) done=${done_n}/${NUM_SHARDS}"
  if [[ "${done_n}" -ge "${NUM_SHARDS}" ]]; then
    bash "${SCRIPT_DIR}/merge-and-validate.sh"
    echo "PASS: all shards merged"
    exit 0
  fi
  NO_WATCH=1 bash "${SCRIPT_DIR}/submit-shards.sh" all || true
  sleep 20
done
