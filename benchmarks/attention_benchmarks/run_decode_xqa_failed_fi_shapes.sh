#!/usr/bin/env bash
# Run XQA-only q1 decode checks for shapes where native FI decode failed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/artifacts/sm120_decode_random_kv_audit}"
OUT_DIR="${OUT_DIR:-${EXPERIMENT_DIR}/raw/xqa_only_failed_fi}"

mkdir -p "${OUT_DIR}/json" "${OUT_DIR}/csv" "${OUT_DIR}/logs"

specs=(
  8q1s64k
  16q1s32k 16q1s64k
  32q1s16k 32q1s32k 32q1s64k
  64q1s8k 64q1s16k 64q1s32k 64q1s64k
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
      --attention-kernels xqa_decode_causal \
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
