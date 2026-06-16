#!/usr/bin/env bash
# Run FlashInfer native non-causal prefill one shape per container.
#
# This isolates CUDA illegal-memory-access failures so a bad shape does not
# destroy the rest of the grid.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/artifacts/sm120_specdec_kernel_microbench}"
OUT_DIR="${OUT_DIR:-${EXPERIMENT_DIR}/raw/fi_noncausal_one_by_one}"

mkdir -p "${OUT_DIR}/json" "${OUT_DIR}/csv" "${OUT_DIR}/logs"

specs=(
  q8s1k q8s2k q8s4k q8s8k q8s16k q8s32k q8s64k
  2q8s1k 2q8s2k 2q8s4k 2q8s8k 2q8s16k 2q8s32k 2q8s64k
  4q8s1k 4q8s2k 4q8s4k 4q8s8k 4q8s16k 4q8s32k 4q8s64k
  8q8s1k 8q8s2k 8q8s4k 8q8s8k 8q8s16k 8q8s32k 8q8s64k
  16q8s1k 16q8s2k 16q8s4k 16q8s8k 16q8s16k 16q8s32k 16q8s64k
  32q8s1k 32q8s2k 32q8s4k 32q8s8k 32q8s16k 32q8s32k 32q8s64k
  64q8s1k 64q8s2k 64q8s4k 64q8s8k 64q8s16k 64q8s32k 64q8s64k
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
      --config configs/sm120_specdec_kernel_microbench_nocg.yaml \
      --attention-kernels triton fi_prefill_noncausal \
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
