# DP Attention Rebalancing Strategy for Agentic/RL Workloads

## Executive Summary

We benchmarked the attention block of Qwen3-235B-A22B (TP4 DP2, 8x H200) under
RL/agentic workload conditions and found three distinct sources of DP imbalance,
each with different magnitudes and solutions. The largest (prefill vs decode mismatch)
wastes up to 2.7 seconds per forward pass at 64k ISL. This document proposes a
multi-layered rebalancing strategy informed by kernel-level measurements.

## Measured Imbalance Sources

### 1. Prefill vs decode mismatch (largest: 122 ms - 2.7 s per forward)

When one DP rank receives a new request (prefill) while the other only decodes,
the prefill rank is dramatically slower:

| Scenario (per layer) | Prefill rank | Decode rank | Gap | x 94 layers |
|---|---|---|---|---|
| 1 req x 2k ISL + 19 decode | 596 us | 124 us | 472 us | 44 ms |
| 1 req x 4k ISL + 19 decode | 830 us | 124 us | 706 us | 66 ms |
| 1 req x 8k ISL + 19 decode | 1,426 us | 124 us | 1,303 us | **122 ms** |
| 3 reqs x 4k ISL + 17 decode | 1,672 us | 124 us | 1,548 us | 145 ms |

Prefill attention is O(ISL^2) per request. A single 64k-ISL request
(extrapolated) would cost ~29 ms/layer, wasting ~2.7 seconds while the
decode-only rank sits idle after finishing in ~12 ms.

**Context wait eliminates this entirely**: when both ranks prefill simultaneously,
the gap drops to < 2 us. TRT-LLM reports 33% throughput improvement with this
strategy on DeepSeek V3.

### 2. Decode KV imbalance (moderate: 11 ms per forward at extreme)

When both ranks decode but one has more total KV tokens:

| Light rank (200k KV) | Heavy rank (1.2M KV) | Gap | x 94 layers |
|---|---|---|---|
| 75 us/layer | 200 us/layer | 125 us | 11.4 ms |

Decode attention scales linearly with total KV (~24 us per 200k tokens on H200).
Per-request KV distribution is irrelevant -- only the sum matters. FA3's split-K
decomposition distributes work across CTAs proportionally to total KV regardless
of how it's split across requests.

### 3. Prefill ISL distribution (moderate: within single-rank scheduling)

For the same total prefill tokens, long requests cost quadratically more:

| Config (total_q = 8k) | Attention | Ratio vs 8x1k |
|---|---|---|
| 1 x 8k ISL | 395 us | 4.3x |
| 2 x 4k ISL | 221 us | 2.4x |
| 4 x 2k ISL | 134 us | 1.5x |
| 8 x 1k ISL | 91 us | 1.0x |

This means balancing total prefill tokens across ranks is not enough.
The optimization target is **sum(ISL_i^2)**, not sum(ISL_i).

## Component Cost Reference

All measurements: Qwen3-235B TP4, 8x H200, CUDA graphs, FLASH_ATTN, bf16, median.

| Component | Decode (per layer) | Prefill (per layer, 8k total_q) |
|---|---|---|
| QKV projection | 12 us (flat) | 200 us (linear in total_q) |
| Attention kernel | 39-160 us (linear in total KV) | 91-395 us (O(ISL^2) per req) |
| O projection | 10 us (flat) | 166 us (linear in total_q) |
| TP all-reduce | 18 us (flat) | 318 us (linear in total_q) |

Projections and allreduce scale linearly with total query tokens. Only attention
has non-linear scaling, and only for prefill. For rebalancing decisions, attention
is the only component that creates meaningful imbalance.

## Proposed Strategies

### Strategy A: Coordinated prefill scheduling (context wait)

**What**: Delay prefill admission on one DP rank until the other rank also has
pending prefill work, then execute both simultaneously.

**Why it works**: Our data shows the gap drops from 1303 us to 1.7 us when
both ranks prefill together. The cost is increased TTFT for the waiting request.

**Optimization target**: Minimize max(sum(q_i * kv_i)) across DP ranks per
iteration, where q_i = ISL for prefill requests and q_i = 1 for decode.

**Implementation considerations**:
- `timeout_iters` parameter (TRT-LLM uses 50): how long to wait before giving
  up and prefilling unilaterally. Too long hurts TTFT; too short loses the benefit.
- `batching_wait_iters` (TRT-LLM uses 10): additional wait to accumulate
  similar-sized prefill batches across ranks, reducing residual imbalance from
  unequal batch accumulation.
- Prefill ISL matching: when both ranks have prefill work, prefer to schedule
  requests that equalize sum(ISL_i^2) across ranks, not just total tokens.

**Expected benefit**: 33% throughput improvement (TRT-LLM's measured result on
DeepSeek V3). Our data suggests the benefit could be even larger for long-ISL
RL workloads (64k contexts).

### Strategy B: Decode KV rebalancing via request migration

**What**: Periodically check total KV tokens per DP rank. If the imbalance
exceeds a threshold, migrate one or more requests (their KV cache blocks) from
the heavy rank to the light rank.

**Why it works**: Decode attention is purely linear in total KV. Moving a request
with 30k KV tokens from the heavy rank reduces its attention time by ~3.6 us/layer
(24 us per 200k tokens * 30k/200k) and increases the light rank by the same amount.

**Optimization target**: Balance sum(kv_len_i) across DP ranks.

**Implementation considerations**:
- KV cache blocks are on specific GPUs. Migration requires either:
  (a) Send/recv of KV blocks between DP ranks (DP=2: point-to-point), or
  (b) Re-prefill the request on the target rank (wasteful for long contexts).
- With TP4, each GPU stores only 1 KV head x 128 dim per block. A 30k-token
  request = 1875 blocks x 16 tokens x 128 dim x 2 (K+V) x 2 bytes = 15 MB per
  GPU. NVLink transfer at ~450 GB/s takes ~33 us -- negligible vs the 11 ms/fwd
  imbalance saved.
- Threshold: trigger when `abs(total_kv_dp0 - total_kv_dp1) > X`. At 24 us per
  200k KV per layer x 94 layers, a 200k imbalance costs 2.3 ms/fwd. A reasonable
  threshold is 100-200k KV token difference.

**Expected benefit**: Up to 11 ms/fwd at extreme imbalance (1.2M vs 200k KV).
Typical benefit in RL workloads: 2-5 ms/fwd.

### Strategy C: Agentic-aware sticky allocation with prefill shuffling

**What**: In agentic/RL workloads, requests repeatedly call tools and wait for
responses. This creates a pattern:
1. Request generates tool call (decode) -> waits for tool response
2. Tool response arrives -> triggers prefill (ISL = tool output length)
3. Request continues decoding

The key insight is that a request on rank X will eventually need a prefill on
rank X (when its tool call returns). Assigning more requests to one rank not only
increases its current decode KV load, but also increases the probability of a
future prefill landing on that rank.

**Proposed approach**:
- **Sticky allocation**: Keep requests on their assigned DP rank to maximize KV
  reuse. Moving a request between tool calls would force a full re-prefill of its
  context on the new rank.
- **Prefill shuffling**: When multiple prefills arrive on the same rank in one
  iteration, temporarily migrate some to the other rank for balanced prefill
  execution. After prefill completes, migrate back. The prefill creates fresh KV
  on the temporary rank, so there's no wasted re-prefill -- but the subsequent
  decode iterations need the KV back on the home rank.
- **Cost**: Each shuffle requires migrating the newly-created KV blocks back
  after prefill. For a 4k-ISL prefill: 250 blocks x 16 x 128 x 2 x 2 = 2 MB,
  transfer time ~4 us at NVLink bandwidth. This is negligible vs the 706 us/layer
  saved by balanced prefill.

**When it helps**: Bursty tool-call returns where multiple requests on the same
rank simultaneously need prefill. Without shuffling, one rank would do 3x4k
prefill (1672 us/layer) while the other decodes (124 us/layer) = 1548 us gap.
With shuffling, each rank does ~1.5x4k prefill ≈ balanced.

### Strategy D: KV freeze instead of eviction

**What**: When a request needs to be preempted (to free KV cache space for
higher-priority work), instead of fully evicting its KV blocks, mark them as
"frozen" -- still allocated but overwritable. If a new request finishes and
frees KV before the frozen blocks are overwritten, the frozen request can resume
without re-prefill.

**Why it helps in RL/max-throughput**: In RL workloads at near-max capacity
(~1.36M tokens used out of 1.36M capacity), preemption is common. A full
eviction of a 60k-token request followed by re-admission requires a 60k-token
prefill costing ~29 ms/layer x 94 layers = ~2.7 seconds. If the request can
be frozen instead, the resume cost is zero (just flip the allocation bit).

**Implementation sketch**:
- KV blocks have a state: `active`, `frozen`, `free`. Frozen blocks are in a
  priority queue ordered by freeze time.
- When allocating blocks for a new request, first use `free` blocks. If none
  available, reclaim the oldest `frozen` blocks.
- When a frozen request resumes: if its blocks are still intact, resume decoding
  at zero cost. If some blocks were reclaimed, re-prefill only the missing suffix
  (partial re-prefill).
- This is essentially an LRU cache on top of the block allocator. vLLM already
  has prefix caching and block recycling; this extends it with explicit freeze
  semantics.

**Expected benefit**: Eliminates the 2.7-second re-prefill cost for preempted
requests in the common case where capacity pressure is temporary. The worst case
(all frozen blocks reclaimed) falls back to full re-prefill, same as today.

## Priority Ranking

| Strategy | Complexity | Benefit per forward | When it helps |
|---|---|---|---|
| A: Context wait | Low | 44-145 ms (1 prefill), up to seconds (64k) | Every prefill event |
| D: KV freeze | Medium | 0-2.7 s (avoids re-prefill) | Preemption under memory pressure |
| B: Decode rebalancing | Medium | 2-11 ms | Sustained KV imbalance |
| C: Prefill shuffling | High | 44-145 ms (bursty prefills) | Multiple tool returns on same rank |

Strategy A has the best effort-to-benefit ratio and should be implemented first.
Strategy D is a natural extension of vLLM's existing block manager.
Strategy B matters at scale but the benefit is smaller.
Strategy C is the most complex and only helps in specific bursty scenarios.

## Unified Cost Model

All strategies optimize the same underlying metric per DP rank per iteration:

```
attention_cost(rank) = sum_over_requests(q_len_i * kv_len_i)

where:
  prefill request: q_len = ISL,   kv_len = ISL          -> cost = ISL^2
  decode request:  q_len = 1,     kv_len = context_len   -> cost = context_len
  extend/chunk:    q_len = chunk, kv_len = accumulated   -> cost = chunk * accumulated
```

The iteration time is bounded by max(attention_cost(rank_0), attention_cost(rank_1)).
All strategies aim to minimize this max by redistributing work across ranks.
