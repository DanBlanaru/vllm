#!/usr/bin/env python3
"""
Grid runner for the DP attention benchmark.

Spawns a separate torchrun process for each (num_reqs, total_kv)
configuration so that vLLM global state is fully reset between configs.
Both DP groups run the same config (symmetric) to measure absolute
component timings across the (num_reqs, total_kv) space.

Usage (inside the container):
    python benchmarks/dp_imbalance/run_grid.py
    python benchmarks/dp_imbalance/run_grid.py --trials 50 --nproc 8
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_SCRIPT = os.path.join(SCRIPT_DIR, "benchmark_dp_attn.py")

COMPONENTS = ["qkv_proj", "attn", "o_proj", "allreduce", "total"]

GRID_GROUPS = [
    ("Batch size sweep (total_kv = 100k)", [
        ( 1, 100_000),
        ( 5, 100_000),
        (10, 100_000),
        (15, 100_000),
        (30, 100_000),
        (60, 100_000),
    ]),
    ("KV cache sweep (num_reqs = 30)", [
        (30,  25_000),
        (30,  50_000),
        (30, 100_000),
        (30, 150_000),
        (30, 200_000),
    ]),
    ("Cross-validation points", [
        (15,  50_000),
        (15, 200_000),
        (60,  50_000),
    ]),
]


def _unique_configs():
    """Deduplicate grid configs preserving order."""
    return list(dict.fromkeys(
        c for _, cfgs in GRID_GROUPS for c in cfgs))


def _run_one(num_reqs, total_kv, args):
    """Run one symmetric benchmark config via torchrun. Returns parsed result dict or None."""
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


def _print_grid(grid_results):
    """Print organized grid results grouped by sweep type."""
    W = 10
    hdr = (f"  {'reqs':>6s}  {'total_kv':>8s}  {'/req':>6s}"
           + "".join(f"{c:>{W}s}" for c in COMPONENTS))
    sep = "  " + "\u2500" * (6 + 2 + 8 + 2 + 6 + W * len(COMPONENTS))

    for group_name, configs in GRID_GROUPS:
        print(f"\n  {group_name}")
        print(hdr)
        print(sep)
        for nr, tkv in configs:
            key = (nr, tkv)
            if key not in grid_results:
                continue
            avg = grid_results[key]
            per_req = tkv // nr
            cells = "".join(f"{avg[c]:>{W}.1f}" for c in COMPONENTS)
            print(f"  {nr:>6d}  {tkv:>8d}  {per_req:>6d}{cells}")
        print(sep)

    print("\n  All times in us (mean across 8 GPUs, symmetric DP0=DP1).")


def main():
    parser = argparse.ArgumentParser(
        description="Grid runner for DP attention benchmark")
    parser.add_argument("--backend", default="FLASH_ATTN")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    configs = _unique_configs()
    print(f"Grid: {len(configs)} configs, {args.trials} trials each, "
          f"backend={args.backend}")
    print(f"Each config runs as a separate torchrun process.\n")

    grid_results = {}
    for i, (nr, tkv) in enumerate(configs):
        per_req = tkv // nr
        cfg_str = (f"{nr:>3d} reqs x {tkv // 1000:>3d}k KV "
                   f"({per_req}/req)")
        tag = f"[{i + 1}/{len(configs)}] DP0: {cfg_str}  DP1: {cfg_str}"
        print(f"  {tag} ...", end=" ", flush=True)

        avg = _run_one(nr, tkv, args)
        if avg is not None:
            grid_results[(nr, tkv)] = avg
            print(f"total={avg['total']:.1f} us")
        else:
            print("FAILED")

    print("\n" + "=" * 80)
    _print_grid(grid_results)
    print("=" * 80)


if __name__ == "__main__":
    main()
