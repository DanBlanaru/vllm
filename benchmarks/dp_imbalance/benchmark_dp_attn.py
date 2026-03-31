#!/usr/bin/env python3
"""
E2E multi-GPU benchmark of the attention block under DP-imbalanced decode.

Problem statement:
  In agentic / RL tool-calling workloads (e.g. Qwen3-235B), a request that
  reads a large file can grow its KV cache to 60k+ tokens while other requests
  stay short. With DP2 TP4, this creates an imbalance where one DP rank has
  far more total KV tokens than the other. We want to measure the actual E2E
  attention layer runtime gap between the two DP ranks and understand which
  components (QKV projection, attention kernel, output projection, TP
  all-reduce) contribute to the imbalance.

Setup:
  DP0 (GPUs 0-3, TP group): 15 requests, ~120k total KV tokens
  DP1 (GPUs 4-7, TP group): 30 requests, ~50k total KV tokens

KV cache setup:
  The KV cache is allocated as zeros. The metadata sets seq_lens = kv_lens
  so the attention kernel reads kv_lens[i] tokens per request from the cache.
  This gives accurate decode memory-bandwidth timing regardless of data
  content. Each call to impl.forward also writes 1 new K/V token per request
  into the cache (negligible cost relative to the reads).

Usage:
    torchrun --nproc_per_node=8 benchmark_dp_attn.py
    torchrun --nproc_per_node=8 benchmark_dp_attn.py --distribution skewed
    torchrun --nproc_per_node=8 benchmark_dp_attn.py --kv-cache-dtype fp8
"""

import argparse
import json
import logging
import os
import sys
import types
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_DIR = os.path.join(SCRIPT_DIR, os.pardir, os.pardir)
ATTN_BENCH_DIR = os.path.join(VLLM_DIR, "benchmarks", "attention_benchmarks")
sys.path.insert(0, os.path.abspath(VLLM_DIR))
sys.path.insert(0, os.path.abspath(ATTN_BENCH_DIR))


# =====================================================================
# Config
# =====================================================================

@dataclass
class ModelConfig:
    name: str
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    tp_size: int

    @property
    def qkv_out_dim(self):
        return (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim

    @property
    def q_out_dim(self):
        return self.num_q_heads * self.head_dim


QWEN3_235B_TP4 = ModelConfig(
    name="Qwen3-235B-A22B TP4",
    hidden_size=4096, num_q_heads=16, num_kv_heads=1,
    head_dim=128, num_layers=94, tp_size=4,
)


@dataclass
class DPScenario:
    label: str
    num_reqs: int
    kv_lens: list

    @property
    def total_kv(self):
        return sum(self.kv_lens)

    @property
    def avg_kv(self):
        return self.total_kv / self.num_reqs if self.num_reqs else 0


def get_scenarios(distribution):
    if distribution == "uniform":
        return (
            DPScenario("DP0", 15, [8192] * 15),
            DPScenario("DP1", 30, [1700] * 30),
        )
    elif distribution == "skewed":
        return (
            DPScenario("DP0", 15, [61440] + [4286] * 14),
            DPScenario("DP1", 30, [5120] * 5 + [1024] * 25),
        )
    raise ValueError(f"Unknown distribution: {distribution}")


def _make_scenario(num_reqs, total_kv, label=""):
    """Create a uniform DPScenario from (num_reqs, total_kv)."""
    per_req = total_kv // num_reqs
    return DPScenario(label or f"{num_reqs}rq", num_reqs, [per_req] * num_reqs)


# =====================================================================
# Timing
# =====================================================================

def time_cuda_us(fn, warmup=10, trials=50, sync_group=None):
    """Returns (mean_us, std_us).

    Args:
        sync_group: If provided, a dist.barrier on this process group is
            inserted before each trial's start event.  This prevents
            collective-op measurements (all_reduce) from including the
            skew caused by GPUs reaching the collective at different
            times after their previous cuda-synchronize.
    """
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if sync_group is not None:
        dist.barrier(group=sync_group)
    times = []
    for _ in range(trials):
        if sync_group is not None:
            dist.barrier(group=sync_group)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


# =====================================================================
# vLLM attention setup
# =====================================================================

def _make_vllm_config(cfg, max_kv, max_num_blocks, kv_cache_dtype):
    from vllm.config import (
        CacheConfig, CompilationConfig, DeviceConfig, LoadConfig,
        ModelConfig as VllmModelConfig, ParallelConfig, SchedulerConfig,
        VllmConfig,
    )
    mc = VllmModelConfig(
        model="meta-llama/Meta-Llama-3-8B",
        tokenizer="meta-llama/Meta-Llama-3-8B",
        trust_remote_code=False, dtype="auto", seed=0,
        max_model_len=max(max_kv, 1024),
    )
    cc = CacheConfig(block_size=16, cache_dtype=kv_cache_dtype)
    cc.num_gpu_blocks = max_num_blocks
    cc.num_cpu_blocks = 0

    mc.get_num_layers = types.MethodType(lambda s: 1, mc)
    mc.get_sliding_window_for_layer = types.MethodType(lambda s, i: None, mc)
    mc.get_logits_soft_cap_for_layer = types.MethodType(lambda s, i: 0.0, mc)
    mc.get_sm_scale_for_layer = types.MethodType(
        lambda s, i, hd=cfg.head_dim: 1.0 / hd**0.5, mc)
    mc.get_num_attention_heads = types.MethodType(
        lambda s, p=None, v=cfg.num_q_heads: v, mc)
    mc.get_num_kv_heads = types.MethodType(
        lambda s, p=None, v=cfg.num_kv_heads: v, mc)
    mc.get_head_size = types.MethodType(lambda s, v=cfg.head_dim: v, mc)
    mc.get_sliding_window = types.MethodType(lambda s: None, mc)

    return VllmConfig(
        model_config=mc, cache_config=cc,
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        scheduler_config=SchedulerConfig(
            max_num_seqs=256,
            max_num_batched_tokens=max(max_kv, 8192),
            max_model_len=max(max_kv, 8192),
            is_encoder_decoder=False, enable_chunked_prefill=True,
        ),
        device_config=DeviceConfig(), load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )


def setup_attention(cfg, scenario, backend_name, kv_cache_dtype, device):
    """
    Set up vLLM AttentionImpl for decode.

    The KV cache is zero-filled. seq_lens = kv_lens so the kernel reads
    kv_lens[i] tokens per request -- accurate decode bandwidth cost.
    """
    from runner import (
        _build_common_attn_metadata, _create_backend_impl,
        _create_kv_cache, _create_metadata_builder, _get_backend_config,
        log_warnings_and_errors_only,
    )
    from common import BenchmarkConfig
    from vllm.config import set_current_vllm_config
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    q_lens = [1] * scenario.num_reqs
    max_kv = max(scenario.kv_lens)
    block_size = 16
    max_blk = (max_kv + block_size - 1) // block_size
    total_blocks = scenario.num_reqs * max_blk

    bench_cfg = BenchmarkConfig(
        backend=backend_name,
        batch_spec=f"{scenario.num_reqs}q1s{max_kv}",
        num_layers=1, head_dim=cfg.head_dim,
        num_q_heads=cfg.num_q_heads, num_kv_heads=cfg.num_kv_heads,
        block_size=block_size, device=str(device),
        kv_cache_dtype=kv_cache_dtype,
    )

    bcfg = _get_backend_config(backend_name)
    with log_warnings_and_errors_only():
        vcfg = _make_vllm_config(cfg, max_kv, total_blocks, kv_cache_dtype)
        dtype = vcfg.model_config.dtype
        with set_current_vllm_config(vcfg):
            backend_class, impl, layer = _create_backend_impl(
                bcfg, bench_cfg, device, dtype)
            from vllm.v1.attention.backends.utils import (
                get_kv_cache_layout, set_kv_cache_layout)
            rl = backend_class.get_required_kv_cache_layout()
            if rl is not None:
                set_kv_cache_layout(rl)
                get_kv_cache_layout.cache_clear()
            meta = _build_common_attn_metadata(
                q_lens, scenario.kv_lens, block_size, device)
            spec = FullAttentionSpec(
                block_size=block_size, num_kv_heads=cfg.num_kv_heads,
                head_size=cfg.head_dim, dtype=dtype)
            builder = _create_metadata_builder(
                backend_class, spec, vcfg, device, backend_name)
            attn_meta = builder.build(
                common_prefix_len=0, common_attn_metadata=meta)
            kv_cache = _create_kv_cache(
                bench_cfg, total_blocks, backend_class, device, dtype)[0]

    return impl, layer, kv_cache, attn_meta, dtype, vcfg


# =====================================================================
# Main benchmark
# =====================================================================

COMPONENTS = ["qkv_proj", "attn", "o_proj", "allreduce", "total"]


def _benchmark_scenario(cfg, scenario, args, device, tp_group):
    """Benchmark one scenario on this GPU. Returns results dict."""
    from vllm.config import set_current_vllm_config

    dtype = torch.bfloat16
    n = scenario.num_reqs
    q_dim = cfg.q_out_dim
    kv_dim = cfg.num_kv_heads * cfg.head_dim

    qkv_w = nn.Linear(cfg.hidden_size, cfg.qkv_out_dim,
                       bias=False, dtype=dtype, device=device)
    o_w = nn.Linear(q_dim, cfg.hidden_size,
                     bias=False, dtype=dtype, device=device)
    hidden = torch.randn(n, cfg.hidden_size, device=device, dtype=dtype)

    impl, layer, kv_cache, attn_meta, mdtype, vcfg = setup_attention(
        cfg, scenario, args.backend, args.kv_cache_dtype, device)

    attn_out = torch.empty(n, cfg.num_q_heads, cfg.head_dim,
                           device=device, dtype=mdtype)
    ar_buf = torch.empty(n, cfg.hidden_size, device=device, dtype=dtype)

    qkv_raw = torch.nn.functional.linear(hidden, qkv_w.weight)
    q_bench = qkv_raw[:, :q_dim].view(
        n, cfg.num_q_heads, cfg.head_dim).clone()
    k_bench = qkv_raw[:, q_dim:q_dim + kv_dim].view(
        n, cfg.num_kv_heads, cfg.head_dim).clone()
    v_bench = qkv_raw[:, q_dim + kv_dim:].view(
        n, cfg.num_kv_heads, cfg.head_dim).clone()
    del qkv_raw

    def fn_qkv():
        torch.nn.functional.linear(hidden, qkv_w.weight)

    def fn_attn():
        impl.forward(layer, q_bench, k_bench, v_bench,
                     kv_cache, attn_meta, output=attn_out)

    def fn_o_proj():
        torch.nn.functional.linear(attn_out.view(n, q_dim), o_w.weight)

    def fn_allreduce():
        dist.all_reduce(ar_buf, group=tp_group)

    def fn_total():
        qkv = torch.nn.functional.linear(hidden, qkv_w.weight)
        q = qkv[:, :q_dim].view(n, cfg.num_q_heads, cfg.head_dim)
        k = qkv[:, q_dim:q_dim + kv_dim].view(
            n, cfg.num_kv_heads, cfg.head_dim)
        v = qkv[:, q_dim + kv_dim:].view(
            n, cfg.num_kv_heads, cfg.head_dim)
        impl.forward(layer, q, k, v, kv_cache, attn_meta, output=attn_out)
        out = torch.nn.functional.linear(
            attn_out.view(n, q_dim), o_w.weight)
        dist.all_reduce(out, group=tp_group)

    with set_current_vllm_config(vcfg):
        for _ in range(args.warmup):
            fn_total()
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)

    with set_current_vllm_config(vcfg):
        results = {
            "qkv_proj":  time_cuda_us(fn_qkv,       5, args.trials),
            "attn":      time_cuda_us(fn_attn,       5, args.trials),
            "o_proj":    time_cuda_us(fn_o_proj,     5, args.trials),
            "allreduce": time_cuda_us(fn_allreduce,  5, args.trials,
                                      sync_group=tp_group),
            "total":     time_cuda_us(fn_total,      5, args.trials,
                                      sync_group=tp_group),
        }

    return results


# =====================================================================
# Output formatting
# =====================================================================

def _print_header(cfg, args, world_size, dp_size, tp_size):
    """Print common benchmark header."""
    print(f"\n{'=' * 90}")
    print(f"  {cfg.name}  |  DP Imbalance Attention Benchmark")
    print(f"{'=' * 90}")
    print(f"  Layout : {world_size} GPUs = {dp_size} DP x {tp_size} TP")
    print(f"  Backend: {args.backend}   KV dtype: {args.kv_cache_dtype}"
          f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Per-GPU: Q={cfg.num_q_heads}h  KV={cfg.num_kv_heads}h"
          f"  d={cfg.head_dim}  hidden={cfg.hidden_size}")
    qkv_mb = cfg.hidden_size * cfg.qkv_out_dim * 2 / 1e6
    o_mb = cfg.q_out_dim * cfg.hidden_size * 2 / 1e6
    print(f"  Weights: QKV [{cfg.hidden_size},{cfg.qkv_out_dim}]"
          f" = {qkv_mb:.1f}MB   O [{cfg.q_out_dim},{cfg.hidden_size}]"
          f" = {o_mb:.1f}MB  (bf16)")


def _print_pair_results(all_results, dp0_sc, dp1_sc, cfg, tp_size,
                        dp_size, world_size):
    """Print per-GPU table and DP summary for a pair run."""
    W = 14
    hdr_cells = "".join(f"{c:>{W}s}" for c in COMPONENTS)
    hdr = f"  {'':>20s}{hdr_cells}"
    sep = "  " + "─" * (20 + W * len(COMPONENTS))

    print(hdr)
    print(sep)

    dp_sums = {d: {c: [] for c in COMPONENTS} for d in range(dp_size)}

    for gpu in range(world_size):
        dp = gpu // tp_size
        tp = gpu % tp_size
        r = all_results[gpu]
        cells = []
        for c in COMPONENTS:
            m, s = r[c]
            cells.append(f"{m:7.1f}±{s:4.1f}")
            dp_sums[dp][c].append(m)
        row = "".join(f"{cell:>{W}s}" for cell in cells)
        label = f"GPU {gpu}  (DP{dp}/TP{tp})"
        print(f"  {label:<20s}{row}")
        if gpu == tp_size - 1:
            print(sep)

    print(sep)

    for dp in range(dp_size):
        sc = dp0_sc if dp == 0 else dp1_sc
        cells = []
        for c in COMPONENTS:
            avg = np.mean(dp_sums[dp][c])
            cells.append(f"{avg:>9.1f}    ")
        row = "".join(f"{cell:>{W}s}" for cell in cells)
        tag = f"{sc.num_reqs}rq {sc.total_kv // 1000}kKV"
        label = f"DP{dp} MEAN ({tag})"
        print(f"  {label:<20s}{row}")

    dp0_max = max(all_results[t]["total"][0] for t in range(tp_size))
    dp1_max = max(all_results[tp_size + t]["total"][0]
                  for t in range(tp_size))
    gap = abs(dp0_max - dp1_max)
    slower = "DP0" if dp0_max > dp1_max else "DP1"
    faster_val = min(dp0_max, dp1_max)
    pct = gap / faster_val * 100 if faster_val > 0 else 0

    print(f"\n  {slower} is {pct:.1f}% slower  "
          f"({gap:.1f} us/layer x {cfg.num_layers} layers = "
          f"{gap * cfg.num_layers / 1000:.2f} ms/fwd wasted)")


# =====================================================================
# Main benchmark
# =====================================================================

@torch.inference_mode()
def run_benchmark(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_size = args.tp_size
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size

    assert world_size == tp_size * dp_size
    assert dp_size == 2, f"Need exactly 2 DP ranks, got {dp_size}"

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    if rank != 0:
        logging.getLogger("vllm").setLevel(logging.ERROR)

    tp_group = None
    for dp in range(dp_size):
        ranks = list(range(dp * tp_size, (dp + 1) * tp_size))
        g = dist.new_group(ranks)
        if dp == dp_rank:
            tp_group = g

    cfg = QWEN3_235B_TP4

    if rank == 0:
        _print_header(cfg, args, world_size, dp_size, tp_size)

    if args.distribution == "custom":
        dp0_sc = _make_scenario(
            args.dp0_reqs, args.dp0_total_kv, "DP0")
        dp1_sc = _make_scenario(
            args.dp1_reqs, args.dp1_total_kv, "DP1")
    else:
        dp0_sc, dp1_sc = get_scenarios(args.distribution)

    scenario = dp0_sc if dp_rank == 0 else dp1_sc

    if rank == 0:
        print(f"\n  DP0: {dp0_sc.num_reqs} reqs   "
              f"{dp0_sc.total_kv:,} KV tokens  "
              f"(avg {dp0_sc.avg_kv:.0f})")
        print(f"  DP1: {dp1_sc.num_reqs} reqs   "
              f"{dp1_sc.total_kv:,} KV tokens  "
              f"(avg {dp1_sc.avg_kv:.0f})")
        print(f"\n  Warming up + running {args.trials} "
              f"trials per component...\n")

    results = _benchmark_scenario(
        cfg, scenario, args, device, tp_group)

    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)

    if rank == 0:
        _print_pair_results(all_results, dp0_sc, dp1_sc, cfg,
                            tp_size, dp_size, world_size)

        dp0_means = {}
        dp1_means = {}
        for c in COMPONENTS:
            dp0_means[c] = float(np.mean(
                [all_results[t][c][0] for t in range(tp_size)]))
            dp1_means[c] = float(np.mean(
                [all_results[tp_size + t][c][0]
                 for t in range(tp_size)]))
        bench_result = {
            "dp0": dp0_means, "dp1": dp1_means,
            "dp0_config": {"num_reqs": dp0_sc.num_reqs,
                           "total_kv": dp0_sc.total_kv},
            "dp1_config": {"num_reqs": dp1_sc.num_reqs,
                           "total_kv": dp1_sc.total_kv},
        }
        print(f"\nBENCH_RESULT:{json.dumps(bench_result)}")

        print(f"{'=' * 90}\n")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="E2E attention block DP imbalance benchmark")
    parser.add_argument("--backend", default="FLASH_ATTN",
                        choices=["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"])
    parser.add_argument("--kv-cache-dtype", default="auto",
                        choices=["auto", "fp8"])
    parser.add_argument("--distribution", default="uniform",
                        choices=["uniform", "skewed", "custom"])
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--dp0-reqs", type=int, default=15,
                        help="DP0 request count (custom mode)")
    parser.add_argument("--dp0-total-kv", type=int, default=100_000,
                        help="DP0 total KV tokens (custom mode)")
    parser.add_argument("--dp1-reqs", type=int, default=30,
                        help="DP1 request count (custom mode)")
    parser.add_argument("--dp1-total-kv", type=int, default=50_000,
                        help="DP1 total KV tokens (custom mode)")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
