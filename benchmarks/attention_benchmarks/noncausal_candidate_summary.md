
## Reproduction

Assume commands are run from the vLLM repo root:

```bash
cd vllm
```

The exact tables in this note are generated from saved raw artifacts. To regenerate the Markdown/CSV tables from those artifacts:

```bash
python benchmarks/attention_benchmarks/summarize_sm120_grid.py --experiment-dir artifacts/sm120_random_kv_audit
python benchmarks/attention_benchmarks/summarize_sm120_grid.py --experiment-dir artifacts/sm120_decode_random_kv_audit --fi-backend fi_decode_native
```

To rerun the q8 specdec proxy into a fresh folder:

```bash
EXPERIMENT_DIR=artifacts/sm120_random_kv_rerun bash benchmarks/attention_benchmarks/run_specdec_triton_xqa_grid.sh
EXPERIMENT_DIR=artifacts/sm120_random_kv_rerun bash benchmarks/attention_benchmarks/run_fi_noncausal_grid_one_by_one.sh
python benchmarks/attention_benchmarks/summarize_sm120_grid.py --experiment-dir artifacts/sm120_random_kv_rerun
```

To rerun the q1 decode control into a fresh folder:

```bash
EXPERIMENT_DIR=artifacts/sm120_decode_random_kv_rerun bash benchmarks/attention_benchmarks/run_decode_grid_one_by_one.sh
EXPERIMENT_DIR=artifacts/sm120_decode_random_kv_rerun bash benchmarks/attention_benchmarks/run_decode_xqa_failed_fi_shapes.sh
python benchmarks/attention_benchmarks/summarize_sm120_grid.py --experiment-dir artifacts/sm120_decode_random_kv_rerun --fi-backend fi_decode_native
```

