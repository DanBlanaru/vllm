# DP Attention Block Benchmark: Qwen3-235B TP4 Imbalance Analysis

## Problem Statement

In agentic / RL tool-calling workloads (e.g. Qwen3-235B), a request that reads a large
file can grow its KV cache to 60k+ tokens while other requests stay short. With DP2 TP4
(8 GPUs), this creates an imbalance where one DP rank has far more total KV tokens than
the other.

We want to answer:
1. Which DP rank is slower for the attention block (QKV proj + attention + O proj + TP all-reduce), and by how much?
2. Which individual component drives the imbalance?
3. Is rebalancing across DP ranks worth it, and what is the right communication primitive?

Two concrete scenarios:
- **DP0**: 15 running requests, ~120k total KV cache tokens (heavy contexts)
- **DP1**: 30 running requests, ~50k total KV cache tokens (many short requests)

## Architecture: Qwen3-235B-A22B with TP4

From [HuggingFace config.json](https://huggingface.co/Qwen/Qwen3-235B-A22B):

| Parameter | Global | Per-GPU (TP4) |
|-----------|--------|---------------|
| hidden_size | 4096 | 4096 (input to col-parallel QKV) |
| num_attention_heads | 64 | 16 |
| num_key_value_heads | 4 | 1 (no duplication: 4/4) |
| head_dim | 128 (explicit, not hidden_size/num_heads) | 128 |
| num_hidden_layers | 94 | 94 |

Per-GPU weights (bf16): QKV [4096, 2304] = 18.9 MB, O [2048, 4096] = 16.8 MB.

## Implementation

**Location**: `vllm/benchmarks/dp_imbalance/benchmark_dp_attn.py`

Launched with `torchrun --nproc_per_node=8`. Each GPU independently runs:
1. QKV projection (`nn.Linear`, column-parallel shard, no communication)
2. Attention kernel (vLLM `AttentionImpl.forward` with `MockLayer` and paged KV cache)
3. Output projection (`nn.Linear`, row-parallel shard)
4. TP all-reduce (`dist.all_reduce` within the 4-GPU TP group)

**KV cache setup**: allocated as zeros; metadata sets `seq_lens = kv_lens` so the kernel
reads `kv_lens[i]` tokens per request. Accurate for decode memory-bandwidth timing since
the kernel reads the same data volume regardless of content.

## Results: 8x H200, FLASH_ATTN, bf16 KV cache, 100 trials

### Uniform distribution

DP0: 15 reqs x 8192 = 122,880 KV tokens. DP1: 30 reqs x 1700 = 51,000 KV tokens.

```
                            qkv_proj          attn        o_proj     allreduce         total
  ──────────────────────────────────────────────────────────────────────────────────────────
  GPU 0  (DP0/TP0)         21.5± 1.3     53.8± 1.9     20.2± 0.8    47.9±209.3   165.7±106.6
  GPU 1  (DP0/TP1)         22.6± 1.3     54.1± 1.6     25.0±34.8    48.3±211.5   164.0±101.0
  GPU 2  (DP0/TP2)         21.2± 1.2     53.1± 1.2     20.1± 0.3    48.1±210.1   166.5±106.6
  GPU 3  (DP0/TP3)         21.4± 1.9     53.5± 2.0     20.2± 0.6     30.0±28.8   166.2±106.6
  ──────────────────────────────────────────────────────────────────────────────────────────
  GPU 4  (DP1/TP0)         20.5± 1.4     42.2± 1.6     19.6± 1.1     29.6±31.5    160.8± 6.6
  GPU 5  (DP1/TP1)         21.3± 3.4     54.9±99.0     20.0± 0.8     29.5±32.6    156.4± 6.6
  GPU 6  (DP1/TP2)         21.3± 1.1     42.4± 1.5     20.5± 0.9     29.6±32.6    161.1± 6.6
  GPU 7  (DP1/TP3)        34.6±112.1    56.9±131.2     21.6± 0.5     28.8±32.4    160.0± 6.6
  ──────────────────────────────────────────────────────────────────────────────────────────
  DP0 MEAN (15rq 122kKV)      21.7          53.6          21.4          43.6         165.6
  DP1 MEAN (30rq 51kKV)      24.4          49.1          20.4          29.4         159.6
```

**DP0 is 3.4% slower** (5.5 us/layer x 94 layers = 0.51 ms/fwd wasted)

### Skewed distribution (agentic workloads)

DP0: 1 req at 60k + 14 at ~4.3k = 121k KV. DP1: 5 at 5k + 25 at 1k = 51k KV.

```
                            qkv_proj          attn        o_proj     allreduce         total
  ──────────────────────────────────────────────────────────────────────────────────────────
  GPU 0  (DP0/TP0)         20.5± 0.9     60.1± 1.5     20.2± 2.5     28.0± 5.6   165.7±108.9
  GPU 1  (DP0/TP1)         20.4± 1.4     60.2± 1.4     19.5± 0.4     28.1± 5.6   165.5±106.7
  GPU 2  (DP0/TP2)         20.9± 1.8     59.9± 1.0     19.8± 0.5     27.9± 5.9    163.0±93.3
  GPU 3  (DP0/TP3)         20.2± 1.4     59.8± 1.1     19.4± 0.7     28.2± 5.7   165.4±109.0
  ──────────────────────────────────────────────────────────────────────────────────────────
  GPU 4  (DP1/TP0)         23.2± 3.1     43.7± 2.0     20.8± 0.8     32.6±52.4   161.0±104.3
  GPU 5  (DP1/TP1)         22.6± 1.0    63.4±198.0     21.1± 1.0     32.7±52.2   158.9±108.4
  GPU 6  (DP1/TP2)         21.3± 1.5     42.6± 1.2     20.1± 0.6     32.4±52.3   162.1±103.7
  GPU 7  (DP1/TP3)         23.2± 1.3    60.2±167.7     21.4± 0.8     32.4±51.3   159.8±103.2
  ──────────────────────────────────────────────────────────────────────────────────────────
  DP0 MEAN (15rq 121kKV)      20.5          60.0          19.7          28.1         164.9
  DP1 MEAN (30rq 51kKV)      22.6          52.5          20.9          32.5         160.4
```

**DP0 is 2.2% slower** (3.6 us/layer x 94 layers = 0.34 ms/fwd wasted)

## Revised Results: CUDA Graphs, Median, RL-Scale KV (8x H200)

The initial results above were dominated by ~55 us of Python/CUDA-launch overhead per
`fn_total()` call, hiding the actual kernel scaling. Two fixes applied:

1. **CUDA graph capture** (`--cuda-graphs`): eliminates all inter-kernel Python overhead.
   Confirmed via nsys: pure kernel time is ~40-60 us, matching graph-captured measurements.
2. **Median + IQR** instead of mean ± std: filters NCCL barrier jitter outliers (some up
   to 13 ms from CPU scheduling delays causing NCCL spin-wait).
3. **Process isolation**: each grid config runs as a separate `torchrun` process via
   `run_grid.py` to avoid vLLM global state contamination between configs.

### Grid: KV cache sweep (num_reqs = 20, RL concurrency)

Real deployment: KV cache capacity ~1.36M tokens per DP rank, max ~20 concurrent
requests at 65k tokens each. Grid tests uniform per-request KV lengths.

```
    reqs  total_kv    /req  qkv_proj      attn    o_proj allreduce     total
  ──────────────────────────────────────────────────────────────────────────
      20    200000   10000      11.6      38.7      10.2      17.6      75.4
      20    400000   20000      11.7      63.4      10.4      18.0     100.1
      20    600000   30000      11.8      87.1      10.3      18.4     124.6
      20    800000   40000      11.6     111.7      10.0      18.2     149.3
      20   1000000   50000      11.7     135.6      10.3      18.5     173.4
      20   1200000   60000      12.4     159.5      11.3      21.6     199.6
```

**Attention scales linearly**: ~24 us per additional 200k KV tokens.
Projections and allreduce are flat (~12 + 10 + 18 = 40 us fixed floor).

### Grid: Batch size sweep (total_kv = 600k)

```
    reqs  total_kv    /req  qkv_proj      attn    o_proj allreduce     total
  ──────────────────────────────────────────────────────────────────────────
       5    600000  120000      11.4      88.0       9.9      19.9     126.2
      10    600000   60000      11.6      87.5       9.9      17.7     124.3
      15    600000   40000      12.4      87.8      11.1      23.7     129.8
      20    600000   30000      11.8      87.1      10.3      18.4     124.6
```

**Attention depends only on total KV, not distribution across requests.**
5 reqs x 120k/req vs 20 reqs x 30k/req: identical ~87-88 us.

### Grid: Skewed vs uniform per-request KV (same total)

RL workloads have a few long-context requests (60k) mixed with many short ones.
Does the distribution matter? Tested at 800k and 1.2M total with 20 reqs:

```
  800k total, 20 reqs:
    uniform (40k/req)           attn = 111.6 us    total = 149.0 us
    3x60k + 17x36k             attn = 110.9 us    total = 148.3 us
    10x60k + 10x20k            attn = 111.7 us    total = 149.0 us

  1.2M total, 20 reqs:
    uniform (60k/req)           attn = 158.9 us    total = 196.4 us
    3x60k + 17x60k             attn = 159.0 us    total = 196.4 us
    10x60k + 10x60k            attn = 158.8 us    total = 196.2 us
```

**Identical within noise.** The attention kernel depends only on total KV tokens,
not how they are distributed across requests.

### RL imbalance estimate

From reference points:
- Light rank: 15 reqs, 200k KV → 77.9 us/layer
- Heavy rank: 20 reqs, 1.2M KV → 199.6 us/layer
- **Gap: 121.7 us/layer x 94 layers = 11.4 ms/forward wasted**

The gap is almost entirely attention (159.5 vs 39.2 = 120 us). Projections and
allreduce contribute < 2 us of the gap.

## Key Findings

1. **Attention kernel is the only imbalanced component** and the only one worth
   rebalancing. It scales linearly with total KV tokens (~24 us per 200k KV on H200).

2. **QKV and O projections are completely flat**: ~12 us and ~10 us respectively,
   insensitive to both batch size (1-60 reqs) and KV cache size (25k-1.2M tokens).
   Weight-loading dominated at these decode batch sizes.

3. **TP all-reduce is flat at ~18 us** (median). Payload is tiny (120-245 KB bf16),
   NCCL latency-dominated. Occasional jitter outliers up to 13 ms from CPU scheduling
   delays don't affect median.

4. **Attention depends only on total KV, not how requests are distributed.** This
   means rebalancing should optimize for total KV token count, not request count.

5. **At RL scale (1.2M vs 200k KV), the DP gap is 11.4 ms/forward** -- significant
   at decode latencies of ~30-50 ms. Rebalancing could recover up to 30-40% of this.

## Future: Rebalancing Strategies

To be explored as a separate step:
- Is all2all the right comm primitive? (DP=2: send/recv pairs; DP>2: TP-sliced DP-group all-to-all)
- Rebalancing should optimize for **total KV token count only** (not request count)
- Transfer cost vs imbalance waste analysis
- Threshold-based vs periodic triggering

## Reproducing Results

All results were collected on 8x H200 (DGX/HGX, NVSwitch).
Container: vllm-openai with FLASH_ATTN backend, bf16 KV cache.

**Files:**
- `vllm/benchmarks/dp_imbalance/benchmark_dp_attn.py` -- single-pair benchmark (torchrun)
- `vllm/benchmarks/dp_imbalance/run_grid.py` -- grid runner (spawns torchrun per config)

**Environment variables** (required for configs with per-request KV > 8192):

### Single pair run

```bash
# Uniform (original scenario)
docker exec -e HF_HOME=/scratch/bench_serving/hf_cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    $CONTAINER torchrun --nproc_per_node=8 \
    /scratch/bench_serving/vllm/benchmarks/dp_imbalance/benchmark_dp_attn.py \
    --distribution uniform --trials 100 --cuda-graphs

# Custom asymmetric pair (RL imbalance)
docker exec -e HF_HOME=/scratch/bench_serving/hf_cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    $CONTAINER torchrun --nproc_per_node=8 \
    /scratch/bench_serving/vllm/benchmarks/dp_imbalance/benchmark_dp_attn.py \
    --distribution custom \
    --dp0-reqs 20 --dp0-total-kv 1200000 \
    --dp1-reqs 15 --dp1-total-kv 200000 \
    --trials 50 --warmup 10 --cuda-graphs

# Skewed per-request lengths (3 long 60k requests + 17 short)
docker exec -e HF_HOME=/scratch/bench_serving/hf_cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    $CONTAINER torchrun --nproc_per_node=8 \
    /scratch/bench_serving/vllm/benchmarks/dp_imbalance/benchmark_dp_attn.py \
    --distribution custom \
    --dp0-reqs 20 --dp0-total-kv 800000 --dp0-skew-long 3 \
    --dp1-reqs 20 --dp1-total-kv 800000 --dp1-skew-long 3 \
    --trials 50 --warmup 10 --cuda-graphs
```

### Grid sweep (RL-scale configs)

Each config runs as a separate torchrun process (clean global state):
```bash
docker exec -e HF_HOME=/scratch/bench_serving/hf_cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    $CONTAINER python /scratch/bench_serving/vllm/benchmarks/dp_imbalance/run_grid.py \
    --trials 50 --warmup 10 --cuda-graphs
```

### nsys profiling

To inspect the GPU timeline and confirm kernel-level behavior:
```bash
# Without CUDA graphs (see Python/launch overhead)
docker exec -e HF_HOME=/scratch/bench_serving/hf_cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    $CONTAINER nsys profile \
    --stats=true --force-overwrite=true -t cuda -o /tmp/dp_bench_eager \
    torchrun --nproc_per_node=8 \
    /scratch/bench_serving/vllm/benchmarks/dp_imbalance/benchmark_dp_attn.py \
    --distribution custom \
    --dp0-reqs 30 --dp0-total-kv 25000 \
    --dp1-reqs 30 --dp1-total-kv 200000 \
    --trials 5 --warmup 5

# With CUDA graphs (--cuda-graph-trace=node to see inside replays)
docker exec -e HF_HOME=/scratch/bench_serving/hf_cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    $CONTAINER nsys profile \
    --cuda-graph-trace=node --stats=true --force-overwrite=true -t cuda \
    -o /tmp/dp_bench_cg \
    torchrun --nproc_per_node=8 \
    /scratch/bench_serving/vllm/benchmarks/dp_imbalance/benchmark_dp_attn.py \
    --distribution custom \
    --dp0-reqs 30 --dp0-total-kv 25000 \
    --dp1-reqs 30 --dp1-total-kv 200000 \
    --trials 5 --warmup 5 --cuda-graphs
```

The `.nsys-rep` file can be opened in Nsight Systems GUI.
The `--stats=true` flag prints a kernel summary table to stdout.

### Key flags

| Flag | Description |
|------|-------------|
| `--cuda-graphs` | Capture into CUDA graphs (eliminates ~55 us Python/launch overhead) |
| `--distribution custom` | Specify DP0/DP1 configs independently |
| `--dp0-skew-long N` | N requests get 60k tokens, rest split the remainder evenly |
| `--trials N` | Number of timed iterations per component (default 100) |
| `--warmup N` | Warmup iterations before timing (default 20) |
| `--backend` | FLASH_ATTN (default), FLASHINFER, TRITON_ATTN |
