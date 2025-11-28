# Prefill DP Imbalance Benchmark Extension

## Background

The decode benchmark showed attention scales linearly with total KV and is insensitive
to per-request distribution. For prefill, the story is fundamentally different: attention
compute is O(q_len x kv_len) per request, so for pure prefill (q_len = ISL, kv_len = ISL)
the cost is O(ISL^2). This means:

- **4 reqs x 2k ISL** = 4 x 2k x 2k = **16M attention FLOPs**
- **1 req x 8k ISL** = 1 x 8k x 8k = **64M attention FLOPs**

Same total tokens (8k), but **4x more compute** for the single long request. This is the
opposite of decode where distribution doesn't matter.

TRT-LLM's [ADP Balance Strategy](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog10_ADP_Balance_Strategy.md)
addresses the imbalance when one DP rank does prefill while another does decode. Their
"context wait" delays prefill until all ranks can do it together, achieving 33% throughput
improvement at the cost of TTFT.

## Implementation

Extended `benchmark_dp_attn.py` with prefill support:

- `DPScenario.q_lens`: optional per-request query lengths (None = decode, all 1s)
- `_make_prefill_scenario(prefill_isls, decode_kv_lens)`: creates mixed prefill+decode batches
- All tensor shapes use `total_q = sum(q_lens)` instead of `num_reqs`
- `setup_attention` passes `q_lens` to metadata builder (ragged query batching)

Grid runner: `run_prefill_grid.py` with ISL distribution sweep and prefill-vs-decode pairs.

Total query tokens per DP rank capped at ~8k-16k (vLLM's `max_num_batched_tokens`).

## Results: 8x H200, FLASH_ATTN, bf16, CUDA graphs, 50 trials (median)

### Part 1: ISL distribution -- prefill cost is O(ISL^2)

Same total_q (total query tokens), different splits. Both DP ranks run the same config.

```
  ISL distribution (total_q = 4k)
      config   reqs  total_q  qkv_proj      attn    o_proj allreduce     total
  ────────────────────────────────────────────────────────────────────────────
        1x4k      1     4000     102.2     115.7      88.1     170.9     476.4
        2x2k      2     4000     102.4      72.2      88.1     171.0     433.6
        4x1k      4     4000     102.5      51.5      88.2     171.3     413.0
       8x500      8     4000     102.3      42.1      88.1     171.0     402.8

  ISL distribution (total_q = 8k)
      config   reqs  total_q  qkv_proj      attn    o_proj allreduce     total
  ────────────────────────────────────────────────────────────────────────────
        1x8k      1     8000     199.8     394.7     166.3     317.6    1075.2
        2x4k      2     8000     200.1     221.0     166.3     317.7     901.2
        4x2k      4     8000     199.8     134.0     166.2     317.4     813.7
        8x1k      8     8000     199.7      91.1     166.3     317.6     770.9

  ISL distribution (total_q = 16k)
      config   reqs  total_q  qkv_proj      attn    o_proj allreduce     total
  ────────────────────────────────────────────────────────────────────────────
       1x16k      1    16000     393.8    1487.0     325.6     603.6    2790.3
        2x8k      2    16000     393.8     789.7     324.7     602.6    2097.9
        4x4k      4    16000     393.5     451.2     325.7     602.3    1747.4
        8x2k      8    16000     393.7     268.2     330.5     602.8    1572.8
       16x1k     16    16000     393.6     173.5     327.2     602.6    1482.3
```

**Attention scales as O(ISL^2)**: 1x8k is 4.3x slower than 8x1k for the same total_q.
QKV, O, allreduce are constant (depend only on total_q, not distribution).

For balanced prefill scheduling, the optimization target is:
**minimize max(sum(ISL_i^2) per DP rank)**, not just total tokens.

### Part 2: Prefill vs decode imbalance

One DP rank processes prefill+decode, the other decode only.
19-20 decode requests x 30k KV each (~570-600k total decode KV per rank).

```
  Prefill vs decode imbalance (DP0=prefill+decode, DP1=decode-only)
  Scenario                                         DP0 total   DP1 total   Gap
  ─────────────────────────────────────────────────────────────────────────────
  DP0: 1x2k pfill + 19 dec  vs  DP1: 20 dec         595.8       124.2     471.5 us
  DP0: 1x4k pfill + 19 dec  vs  DP1: 20 dec         829.7       123.6     706.0 us
  DP0: 1x8k pfill + 19 dec  vs  DP1: 20 dec        1425.9       123.2    1302.7 us
  DP0: 3x4k pfill + 17 dec  vs  DP1: 20 dec        1671.7       123.9    1547.7 us
```

A single 8k-ISL prefill makes the DP rank **11.5x slower**. Over 94 layers:
**1302.7 us x 94 = 122 ms/forward wasted**. A 64k prefill would waste ~2.7 seconds.

### Part 3: Balanced prefill (simulated context wait)

Both DP ranks do prefill simultaneously (TRT-LLM's "context wait" strategy):

```
  Balanced prefill (both ranks prefill simultaneously)
  Scenario                                         DP0 total   DP1 total   Gap
  ─────────────────────────────────────────────────────────────────────────────
  Both: 1x4k pfill + 19 dec                         828.5       830.2       1.7 us
  Both: 1x8k pfill + 19 dec                        1425.3      1427.1       1.7 us
```

**Gap drops to noise** (~2 us). Context wait fully eliminates prefill imbalance.

## Key Findings

1. **Prefill attention cost is O(ISL^2)**, unlike decode where it's O(total_kv).
   Distribution across requests matters: 1x8k costs 4.3x more than 8x1k at the
   same total_q. The optimization target for prefill balancing is sum(ISL_i^2),
   not sum(ISL_i).

2. **Prefill vs decode imbalance is massive**: a single 8k-ISL prefill wastes
   122 ms/forward (1303 us/layer x 94 layers). This dwarfs decode imbalance
   (11 ms at worst case).

3. **Context wait eliminates prefill imbalance completely**: when both ranks
   prefill simultaneously, the gap is < 2 us. The cost is increased TTFT.

4. **Projections and allreduce scale linearly with total_q** (not ISL^2):
   QKV ~25 us per 1k tokens, O ~21 us per 1k tokens, allreduce ~40 us per 1k tokens.
   These are balanced as long as total_q is balanced.

5. **Unified optimization metric**: attention cost per request = q_len_i x kv_len_i.
   For prefill: q=kv=ISL, so cost = ISL^2. For decode: q=1, so cost = kv_len.
   Balance sum(q_i * kv_i) across DP ranks.

## Reproducing

```bash
# Prefill grid (ISL distribution + imbalance)
docker exec -e HF_HOME=/scratch/bench_serving/hf_cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    $CONTAINER python /scratch/bench_serving/vllm/benchmarks/dp_imbalance/run_prefill_grid.py \
    --trials 50 --warmup 10 --cuda-graphs --part all

# Single prefill pair
docker exec -e HF_HOME=/scratch/bench_serving/hf_cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    $CONTAINER torchrun --nproc_per_node=8 \
    /scratch/bench_serving/vllm/benchmarks/dp_imbalance/benchmark_dp_attn.py \
    --distribution custom \
    --dp0-prefill-isls 4000,4000 --dp0-decode-kvs 30000,30000,30000 \
    --dp1-prefill-isls 4000,4000 --dp1-decode-kvs 30000,30000,30000 \
    --trials 50 --warmup 10 --cuda-graphs
```
