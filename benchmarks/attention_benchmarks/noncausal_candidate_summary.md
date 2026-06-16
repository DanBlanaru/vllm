
## Reproduction

Update after debugging: the previous q8 FI `EXIT_139` rows were caused by the benchmark harness sizing FlashInfer metadata buffers from a mock `max_model_len=1024`, not by the base FlashInfer paged prefill call. The harness now sizes `max_model_len` from the benchmark shape and allows synthetic long contexts. With that fix, the old q8 FI crash rows run; only `64q8s64k` still OOMs.

Assume commands are run from the vLLM repo root:

```bash
cd vllm
```

The exact tables in this note are generated from saved raw artifacts. To regenerate the Markdown/CSV tables from those artifacts:

```bash
python benchmarks/attention_benchmarks/summarize_sm120_grid.py --experiment-dir artifacts/sm120_random_kv_maxlen_fix
python benchmarks/attention_benchmarks/summarize_sm120_grid.py --experiment-dir artifacts/sm120_decode_random_kv_script_e2e --fi-backend fi_decode_native
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

