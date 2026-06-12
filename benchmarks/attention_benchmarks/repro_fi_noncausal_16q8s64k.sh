#!/usr/bin/env bash
# Reproduce the SM120 FlashInfer native non-causal prefill crash.
#
# Run from anywhere on the bench_serving host:
#   bash vllm-sm120-specdec-kernel-bench/benchmarks/attention_benchmarks/repro_fi_noncausal_16q8s64k.sh
#
# Expected failure:
#   CUDA error: an illegal memory access was encountered
# around the single fi_prefill_noncausal q8 shape with 16 contexts and 64k ISL.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${BENCH_ROOT}/artifacts/sm120_specdec_kernel_microbench}"
OUT_DIR="${EXPERIMENT_DIR}/raw/repros"

mkdir -p "${OUT_DIR}"

cd "${BENCH_ROOT}"

make container-launch-vllm-detach \
  VLLM_DIR=vllm-sm120-specdec-kernel-bench \
  RUN_CMD="bash -lc 'cd ${BENCH_ROOT}/vllm-sm120-specdec-kernel-bench/benchmarks/attention_benchmarks && CUDA_LAUNCH_BLOCKING=1 python benchmark.py --config configs/sm120_specdec_kernel_microbench_nocg.yaml --attention-kernels fi_prefill_noncausal --batch-specs 16q8s64k --repeats 1 --warmup-iters 1 --output-json ${OUT_DIR}/fi_16q8s64k_repro.json --output-csv ${OUT_DIR}/fi_16q8s64k_repro.csv'"
