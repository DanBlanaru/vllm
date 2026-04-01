---
name: Prefill DP Imbalance Benchmark
overview: Extend the DP attention benchmark to support prefill workloads, quantify the cost of prefill/decode imbalance (what TRT-LLM's ADP Balance addresses), and measure how ISL distribution affects runtime for the same total token count.
todos:
  - id: p1
    content: Extend DPScenario with q_lens field and add _make_prefill_scenario helper
    status: pending
  - id: p2
    content: Modify _benchmark_scenario to use total_q for tensor shapes and pass q_lens to setup_attention
    status: pending
  - id: p3
    content: Add prefill CLI args (--dp0-isls, --dp0-decode-kvs, etc.)
    status: pending
  - id: p4
    content: Add prefill grid configs to run_grid.py (ISL distribution + prefill-vs-decode imbalance)
    status: pending
  - id: p5
    content: Run prefill ISL distribution grid
    status: pending
  - id: p6
    content: Run prefill-vs-decode imbalance scenarios and compare with balanced (context wait)
    status: pending
  - id: p7
    content: Update plan document with prefill results
    status: pending
isProject: false
---

# Prefill DP Imbalance Benchmark Extension

## Background

The decode benchmark showed attention scales linearly with total KV and is insensitive to per-request distribution. For prefill, the story is fundamentally different: attention compute is O(q_len x kv_len) per request, so for pure prefill (q_len = ISL, kv_len = ISL) the cost is O(ISL^2). This means:

- **4 reqs x 2k ISL** = 4 x 2k x 2k = **16M attention FLOPs**
- **1 req x 8k ISL** = 1 x 8k x 8k = **64M attention FLOPs**

Same total tokens (8k), but **4x more compute** for the single long request. This is the opposite of decode where distribution doesn't matter.

TRT-LLM's [ADP Balance Strategy](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog10_ADP_Balance_Strategy.md) addresses a different axis: the imbalance when one DP rank does prefill (expensive) while another does decode (cheap). Their "context wait" delays prefill until all ranks can do it together, achieving 33% throughput improvement at the cost of TTFT.

## What We Want to Measure

### Part 1: Prefill vs Decode imbalance (quantify ADP Balance benefit)

One DP rank processes a prefill request while the other does only decode. This is the worst-case imbalance that ADP Balance eliminates.

Example scenarios (with 20 concurrent requests at ~600k total KV per rank):

- **Imbalanced**: DP0 = 1 prefill (ISL=4k) + 19 decode (KV=30k), DP1 = 20 decode (KV=30k)
- **Balanced (context wait)**: DP0 = 1 prefill (ISL=4k) + 19 decode, DP1 = 1 prefill (ISL=4k) + 19 decode
- The gap between these quantifies the benefit of coordinated scheduling.

### Part 2: Prefill ISL distribution effects

For the same total prefill tokens, how does splitting across requests affect runtime?

- 1 x 8k ISL vs 2 x 4k vs 4 x 2k vs 8 x 1k (all = 8k total tokens)
- 1 x 16k vs 4 x 4k vs 16 x 1k (all = 16k total tokens)
- This tests whether FA3/FA4's CTA scheduling can hide the O(ISL^2) cost through parallelism.

## Code Changes

### 1. Extend `DPScenario` and `_make_scenario` in `benchmark_dp_attn.py`

Currently `q_lens` is hardcoded to `[1] * num_reqs` (decode). Add a `q_lens` field:

```python
@dataclass
class DPScenario:
    label: str
    num_reqs: int
    kv_lens: list
    q_lens: list = None  # None -> decode (all 1s)

    @property
    def total_q(self):
        return sum(self.q_lens) if self.q_lens else self.num_reqs
```

Add `_make_prefill_scenario(num_reqs, isl_list, decode_kv_lens=None)`:

- `isl_list`: ISL per prefill request (q_len = kv_len = ISL for pure prefill)
- `decode_kv_lens`: optional list of KV lengths for additional decode requests in the same batch (mixed prefill+decode)

### 2. Modify `_benchmark_scenario` for variable `total_q`

Key shape changes when `total_q != num_reqs`:


| Tensor                          | Decode (`total_q = n`) | Prefill (`total_q >> n`)  |
| ------------------------------- | ---------------------- | ------------------------- |
| `hidden`                        | `[n, hidden_size]`     | `[total_q, hidden_size]`  |
| `q_bench`, `k_bench`, `v_bench` | `[n, heads, dim]`      | `[total_q, heads, dim]`   |
| `attn_out`                      | `[n, q_heads, dim]`    | `[total_q, q_heads, dim]` |
| `ar_buf` / O-proj output        | `[n, hidden_size]`     | `[total_q, hidden_size]`  |


The `setup_attention` function already accepts arbitrary `q_lens` via `_build_common_attn_metadata` -- just pass the scenario's `q_lens` instead of `[1]*n`.

### 3. Add prefill CLI args

- `--dp0-isls`: comma-separated ISL per prefill request (e.g., `"4000,4000,4000"`)
- `--dp0-decode-kvs`: comma-separated KV per decode request in the same batch
- Same for dp1

### 4. Add prefill grid configs in `run_grid.py`

New grid groups:

```
Prefill ISL distribution (total_tokens = 8k)
  1 x 8000 ISL
  2 x 4000 ISL  
  4 x 2000 ISL
  8 x 1000 ISL

Prefill ISL distribution (total_tokens = 16k)
  1 x 16000 ISL
  2 x 8000 ISL
  4 x 4000 ISL
  8 x 2000 ISL
  16 x 1000 ISL

Prefill vs Decode imbalance (20 reqs, ~600k decode KV)
  DP0: 1 prefill (ISL=2k) + 19 decode    vs  DP1: 20 decode
  DP0: 1 prefill (ISL=4k) + 19 decode    vs  DP1: 20 decode
  DP0: 1 prefill (ISL=8k) + 19 decode    vs  DP1: 20 decode
  DP0: 3 prefill (ISL=4k) + 17 decode    vs  DP1: 20 decode

Balanced (simulated context wait)
  DP0: 1 prefill (ISL=4k) + 19 decode    vs  DP1: 1 prefill (ISL=4k) + 19 decode
```

### 5. CUDA graph handling

CUDA graphs work for fixed shapes. Since each grid config runs as a separate `torchrun` process, each captures its own graph with the correct `total_q`. No bucketing needed.

However, for **asymmetric prefill vs decode pairs**, DP0 and DP1 have different `total_q` values. Each GPU captures its own graph with its own shapes, which is fine -- the graph is per-GPU.

## Execution Plan

1. Implement the code changes (scenario, benchmark, CLI, grid)
2. Run prefill-only grid (ISL distribution effects) with `--cuda-graphs`
3. Run prefill-vs-decode imbalance scenarios
4. Compare imbalanced vs balanced (context wait) pairs
5. Update the plan document with results and analysis

