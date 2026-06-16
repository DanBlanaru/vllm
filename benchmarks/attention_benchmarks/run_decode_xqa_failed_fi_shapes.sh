#!/usr/bin/env bash
# Run XQA-only q1 decode checks for shapes where native FI decode failed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${BENCH_ROOT}/artifacts/sm120_decode_random_kv_audit}"
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
  cid="$(
    make -s -C "${BENCH_ROOT}" container-launch-vllm-detach \
      VLLM_DIR=vllm-sm120-specdec-kernel-bench \
      RUN_CMD="bash -lc 'cd ${BENCH_ROOT}/vllm-sm120-specdec-kernel-bench/benchmarks/attention_benchmarks && python benchmark.py --config configs/sm120_decode_kernel_microbench_nocg.yaml --attention-kernels xqa_decode_causal --batch-specs ${spec} --output-json ${json} --output-csv ${csv}'" \
      | tail -n 1
  )"
  echo "container=${cid}" | tee "${log}"
  set +e
  exit_code="$(docker wait "${cid}")"
  wait_status=$?
  docker logs "${cid}" >> "${log}" 2>&1
  set -e

  if [[ "${wait_status}" -ne 0 ]]; then
    status="DOCKER_WAIT_ERR"
    exit_code="${wait_status}"
  elif [[ "${exit_code}" == "0" && -s "${json}" ]]; then
    status="OK"
  else
    status="ERR"
  fi
  printf "%s,%s,%s,%s,%s,%s\n" \
    "${spec}" "${status}" "${cid}" "${exit_code}" "${json}" "${log}" >> "${summary}"
  echo "${spec}: ${status} exit=${exit_code}"
done

echo "Summary: ${summary}"
