#!/usr/bin/env bash
# Run the SM120 q1 decode grid one shape per container.
#
# This isolates CUDA failures so a bad shape does not destroy the rest of the
# non-spec decode comparison.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/artifacts/sm120_decode_random_kv_audit}"
OUT_DIR="${OUT_DIR:-${EXPERIMENT_DIR}/raw/decode_one_by_one}"

mkdir -p "${OUT_DIR}/json" "${OUT_DIR}/csv" "${OUT_DIR}/logs"

specs=(
  q1s1k q1s2k q1s4k q1s8k q1s16k q1s32k q1s64k
  2q1s1k 2q1s2k 2q1s4k 2q1s8k 2q1s16k 2q1s32k 2q1s64k
  4q1s1k 4q1s2k 4q1s4k 4q1s8k 4q1s16k 4q1s32k 4q1s64k
  8q1s1k 8q1s2k 8q1s4k 8q1s8k 8q1s16k 8q1s32k 8q1s64k
  16q1s1k 16q1s2k 16q1s4k 16q1s8k 16q1s16k 16q1s32k 16q1s64k
  32q1s1k 32q1s2k 32q1s4k 32q1s8k 32q1s16k 32q1s32k 32q1s64k
  64q1s1k 64q1s2k 64q1s4k 64q1s8k 64q1s16k 64q1s32k 64q1s64k
)

summary="${OUT_DIR}/summary.csv"
printf "batch_spec,status,container_id,exit_code,json,log\n" > "${summary}"

for spec in "${specs[@]}"; do
  json="${OUT_DIR}/json/${spec}.json"
  csv="${OUT_DIR}/csv/${spec}.csv"
  log="${OUT_DIR}/logs/${spec}.log"
  rm -f "${json}" "${csv}" "${log}"

  echo "=== ${spec} ==="
  set +e
  (
    cd "${SCRIPT_DIR}"
    "${PYTHON}" benchmark.py \
      --config configs/sm120_decode_kernel_microbench_nocg.yaml \
      --batch-specs "${spec}" \
      --output-json "${json}" \
      --output-csv "${csv}"
  ) > "${log}" 2>&1
  exit_code="$?"
  set -e

  if [[ "${exit_code}" == "0" && -s "${json}" ]]; then
    status="OK"
  else
    status="ERR"
  fi
  printf "%s,%s,%s,%s,%s,%s\n" \
    "${spec}" "${status}" "local" "${exit_code}" "${json}" "${log}" >> "${summary}"
  echo "${spec}: ${status} exit=${exit_code}"
done

echo "Summary: ${summary}"
