#!/usr/bin/env bash
# Run the q8 specdec Triton/XQA comparison used by the SM120 non-causal
# candidate summary.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_ROOT}/artifacts/sm120_random_kv_audit}"
OUT_DIR="${OUT_DIR:-${EXPERIMENT_DIR}/raw}"

mkdir -p "${OUT_DIR}"

cd "${SCRIPT_DIR}"

"${PYTHON}" benchmark.py \
  --config configs/sm120_specdec_kernel_microbench_nocg.yaml \
  --attention-kernels triton xqa_decode_causal \
  --output-json "${OUT_DIR}/triton_xqa_nocg.json" \
  --output-csv "${OUT_DIR}/triton_xqa_nocg.csv"

# This cell is intentionally omitted from the main YAML because FI crashes
# there, but Triton/XQA are still useful for the final table.
"${PYTHON}" benchmark.py \
  --config configs/sm120_specdec_kernel_microbench_nocg.yaml \
  --attention-kernels triton xqa_decode_causal \
  --batch-specs 16q8s64k \
  --output-json "${OUT_DIR}/triton_xqa_16q8s64k.json" \
  --output-csv "${OUT_DIR}/triton_xqa_16q8s64k.csv"

# Keep this as an explicit isolated row because it may OOM while still writing
# a benchmark-level error row.
"${PYTHON}" benchmark.py \
  --config configs/sm120_specdec_kernel_microbench_nocg.yaml \
  --attention-kernels triton xqa_decode_causal \
  --batch-specs 64q8s64k \
  --output-json "${OUT_DIR}/triton_xqa_64q8s64k.json" \
  --output-csv "${OUT_DIR}/triton_xqa_64q8s64k.csv"
