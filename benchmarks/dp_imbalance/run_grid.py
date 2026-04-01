#!/usr/bin/env python3
"""
Grid runner for the DP attention benchmark.

Spawns a separate torchrun process for each configuration so that vLLM
global state is fully reset between configs. Both DP groups run the same
config (symmetric) to measure absolute component timings.

Usage (inside the container):
    python benchmarks/dp_imbalance/run_grid.py --cuda-graphs
    python benchmarks/dp_imbalance/run_grid.py --trials 50 --cuda-graphs
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_SCRIPT = os.path.join(SCRIPT_DIR, "benchmark_dp_attn.py")

COMPONENTS = ["qkv_proj", "attn", "o_proj", "allreduce", "total"]

# Each entry: (num_reqs, total_kv, skew_long)
#   skew_long = number of requests that get 60k tokens (rest split remainder)
#   skew_long = 0 means uniform distribution

# Grid sized for Qwen3-235B TP4 RL workloads:
#   KV cache capacity ~1.36M tokens per DP rank
#   Max concurrency ~20 requests at 65k tokens each
GRID_GROUPS = [
    ("KV cache sweep -- uniform (num_reqs=20)", [
        (20,  200_000, 0),
        (20,  400_000, 0),
        (20,  600_000, 0),
        (20,  800_000, 0),
        (20, 1_000_000, 0),
        (20, 1_200_000, 0),
    ]),
    ("KV cache sweep -- skewed: 3x60k long (num_reqs=20)", [
        (20,  400_000, 3),
        (20,  600_000, 3),
        (20,  800_000, 3),
        (20, 1_000_000, 3),
        (20, 1_200_000, 3),
    ]),
    ("KV cache sweep -- skewed: 10x60k long (num_reqs=20)", [
        (20,  800_000, 10),
        (20, 1_000_000, 10),
        (20, 1_200_000, 10),
    ]),
    ("Batch size sweep -- uniform (total_kv=600k)", [
        ( 5, 600_000, 0),
        (10, 600_000, 0),
        (15, 600_000, 0),
        (20, 600_000, 0),
    ]),
]


def _unique_configs():
    """Deduplicate grid configs preserving order."""
    return list(dict.fromkeys(
        c for _, cfgs in GRID_GROUPS for c in cfgs))


def _run_one(num_reqs, total_kv, skew_long, args):
    """Run one symmetric benchmark config via torchrun."""
    cmd = [
        "torchrun", f"--nproc_per_node={args.nproc}",
        BENCH_SCRIPT,
        "--distribution", "custom",
        "--dp0-reqs", str(num_reqs),
        "--dp0-total-kv", str(total_kv),
        "--dp1-reqs", str(num_reqs),
        "--dp1-total-kv", str(total_kv),
        "--backend", args.backend,
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--tp-size", str(args.tp_size),
        "--warmup", str(args.warmup),
        "--trials", str(args.trials),
    ]
    if skew_long > 0:
        cmd += ["--dp0-skew-long", str(skew_long),
                "--dp1-skew-long", str(skew_long)]
    if args.cuda_graphs:
        cmd.append("--cuda-graphs")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        sys.stderr.write(f"FAILED (exit {result.returncode})\n")
        sys.stderr.write(result.stderr[-500:] if result.stderr else "")
        return None

    for line in result.stdout.splitlines():
        if line.strip().startswith("BENCH_RESULT:"):
            payload = line.strip()[len("BENCH_RESULT:"):]
            data = json.loads(payload)
            avg = {}
            for c in COMPONENTS:
                avg[c] = (data["dp0"][c] + data["dp1"][c]) / 2.0
            return avg

    sys.stderr.write("  no BENCH_RESULT line found\n")
    return None


def _describe_config(nr, tkv, skew):
    """Human-readable config description."""
    if skew > 0:
        long_len = min(60_000, tkv // skew)
        short_count = nr - skew
        short_len = max(1, (tkv - long_len * skew) // short_count)
        return (f"{nr:>3d} reqs x {tkv // 1000:>4d}k KV "
                f"({skew}x{long_len // 1000}k + "
                f"{short_count}x{short_len // 1000}k)")
    per_req = tkv // nr
    return (f"{nr:>3d} reqs x {tkv // 1000:>4d}k KV "
            f"({per_req}/req)")


def _print_grid(grid_results):
    """Print organized grid results grouped by sweep type."""
    W = 10
    hdr = (f"  {'reqs':>6s}  {'total_kv':>8s}  {'skew':>5s}"
           + "".join(f"{c:>{W}s}" for c in COMPONENTS))
    sep = "  " + "\u2500" * (6 + 2 + 8 + 2 + 5 + W * len(COMPONENTS))

    for group_name, configs in GRID_GROUPS:
        print(f"\n  {group_name}")
        print(hdr)
        print(sep)
        for nr, tkv, skew in configs:
            key = (nr, tkv, skew)
            if key not in grid_results:
                continue
            avg = grid_results[key]
            skew_str = f"{skew}x60k" if skew > 0 else "even"
            cells = "".join(f"{avg[c]:>{W}.1f}" for c in COMPONENTS)
            print(f"  {nr:>6d}  {tkv:>8d}  {skew_str:>5s}{cells}")
        print(sep)

    print("\n  All times in us (median across 8 GPUs, symmetric DP0=DP1).")


def main():
    parser = argparse.ArgumentParser(
        description="Grid runner for DP attention benchmark")
    parser.add_argument("--backend", default="FLASH_ATTN")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="Capture into CUDA graphs")
    args = parser.parse_args()

    configs = _unique_configs()
    print(f"Grid: {len(configs)} configs, {args.trials} trials each, "
          f"backend={args.backend}")
    print(f"Each config runs as a separate torchrun process.\n")

    grid_results = {}
    for i, (nr, tkv, skew) in enumerate(configs):
        cfg_str = _describe_config(nr, tkv, skew)
        tag = f"[{i + 1}/{len(configs)}] DP0: {cfg_str}  DP1: {cfg_str}"
        print(f"  {tag} ...", end=" ", flush=True)

        avg = _run_one(nr, tkv, skew, args)
        if avg is not None:
            grid_results[(nr, tkv, skew)] = avg
            print(f"total={avg['total']:.1f} us")
        else:
            print("FAILED")

    print("\n" + "=" * 80)
    _print_grid(grid_results)
    print("=" * 80)


if __name__ == "__main__":
    main()
