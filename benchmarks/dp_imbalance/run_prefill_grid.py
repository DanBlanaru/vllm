#!/usr/bin/env python3
"""
Prefill grid runner for the DP attention benchmark.

Measures:
  Part 1: ISL distribution effects -- does 1x8k ISL cost more than 4x2k?
  Part 2: Prefill vs decode imbalance -- quantify ADP Balance benefit

Each config runs as a separate torchrun to reset global state.
Total query tokens per DP rank capped at ~8k-16k (vLLM scheduling budget).

Usage (inside the container):
    python benchmarks/dp_imbalance/run_prefill_grid.py --cuda-graphs
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_SCRIPT = os.path.join(SCRIPT_DIR, "benchmark_dp_attn.py")

COMPONENTS = ["qkv_proj", "attn", "o_proj", "allreduce", "total"]

# --- Part 1: ISL distribution (symmetric DP0=DP1, pure prefill) ---
# Same total_q, different splits. Tests O(ISL^2) attention scaling.
PREFILL_DIST_GROUPS = [
    ("ISL distribution (total_q = 8k)", [
        {"isls": [8000],                     "label": "1x8k"},
        {"isls": [4000, 4000],               "label": "2x4k"},
        {"isls": [2000] * 4,                 "label": "4x2k"},
        {"isls": [1000] * 8,                 "label": "8x1k"},
    ]),
    ("ISL distribution (total_q = 16k)", [
        {"isls": [16000],                    "label": "1x16k"},
        {"isls": [8000, 8000],               "label": "2x8k"},
        {"isls": [4000] * 4,                 "label": "4x4k"},
        {"isls": [2000] * 8,                 "label": "8x2k"},
        {"isls": [1000] * 16,                "label": "16x1k"},
    ]),
    ("ISL distribution (total_q = 4k)", [
        {"isls": [4000],                     "label": "1x4k"},
        {"isls": [2000, 2000],               "label": "2x2k"},
        {"isls": [1000] * 4,                 "label": "4x1k"},
        {"isls": [500] * 8,                  "label": "8x500"},
    ]),
]

# --- Part 2: Prefill vs decode imbalance (asymmetric) ---
# DP0 does prefill + decode, DP1 does decode only.
# 19 decode reqs x 30k KV each = 570k decode KV per rank.
_DKV19 = ",".join(["30000"] * 19)
_DKV20 = ",".join(["30000"] * 20)

IMBALANCE_CONFIGS = [
    ("Prefill vs decode imbalance (DP0=prefill+decode, DP1=decode-only)", [
        {"dp0_isls": "2000",  "dp0_dkvs": _DKV19,
         "dp1_isls": None,    "dp1_dkvs": _DKV20,
         "label": "DP0: 1x2k pfill + 19 dec  vs  DP1: 20 dec"},
        {"dp0_isls": "4000",  "dp0_dkvs": _DKV19,
         "dp1_isls": None,    "dp1_dkvs": _DKV20,
         "label": "DP0: 1x4k pfill + 19 dec  vs  DP1: 20 dec"},
        {"dp0_isls": "8000",  "dp0_dkvs": _DKV19,
         "dp1_isls": None,    "dp1_dkvs": _DKV20,
         "label": "DP0: 1x8k pfill + 19 dec  vs  DP1: 20 dec"},
        {"dp0_isls": "4000,4000,4000", "dp0_dkvs": ",".join(["30000"]*17),
         "dp1_isls": None,    "dp1_dkvs": _DKV20,
         "label": "DP0: 3x4k pfill + 17 dec  vs  DP1: 20 dec"},
    ]),
    ("Balanced prefill (simulated context wait)", [
        {"dp0_isls": "4000",  "dp0_dkvs": _DKV19,
         "dp1_isls": "4000",  "dp1_dkvs": _DKV19,
         "label": "DP0: 1x4k pfill + 19 dec  vs  DP1: 1x4k pfill + 19 dec"},
        {"dp0_isls": "8000",  "dp0_dkvs": _DKV19,
         "dp1_isls": "8000",  "dp1_dkvs": _DKV19,
         "label": "DP0: 1x8k pfill + 19 dec  vs  DP1: 1x8k pfill + 19 dec"},
    ]),
]


def _run_symmetric_prefill(isls, args):
    """Run symmetric pure-prefill config (both DP ranks same)."""
    isls_str = ",".join(str(x) for x in isls)
    cmd = [
        "torchrun", f"--nproc_per_node={args.nproc}",
        BENCH_SCRIPT,
        "--distribution", "custom",
        "--dp0-prefill-isls", isls_str,
        "--dp1-prefill-isls", isls_str,
        "--backend", args.backend,
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--tp-size", str(args.tp_size),
        "--warmup", str(args.warmup),
        "--trials", str(args.trials),
    ]
    if args.cuda_graphs:
        cmd.append("--cuda-graphs")
    return _run_cmd(cmd)


def _run_asymmetric(cfg, args):
    """Run asymmetric prefill-vs-decode config."""
    cmd = [
        "torchrun", f"--nproc_per_node={args.nproc}",
        BENCH_SCRIPT,
        "--distribution", "custom",
        "--backend", args.backend,
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--tp-size", str(args.tp_size),
        "--warmup", str(args.warmup),
        "--trials", str(args.trials),
    ]
    if cfg["dp0_isls"]:
        cmd += ["--dp0-prefill-isls", cfg["dp0_isls"]]
    if cfg.get("dp0_dkvs"):
        cmd += ["--dp0-decode-kvs", cfg["dp0_dkvs"]]
    if not cfg["dp0_isls"] and cfg.get("dp0_dkvs"):
        n = len(cfg["dp0_dkvs"].split(","))
        total = sum(int(x) for x in cfg["dp0_dkvs"].split(","))
        cmd += ["--dp0-reqs", str(n), "--dp0-total-kv", str(total)]

    if cfg["dp1_isls"]:
        cmd += ["--dp1-prefill-isls", cfg["dp1_isls"]]
    if cfg.get("dp1_dkvs"):
        cmd += ["--dp1-decode-kvs", cfg["dp1_dkvs"]]
    if not cfg["dp1_isls"] and cfg.get("dp1_dkvs"):
        n = len(cfg["dp1_dkvs"].split(","))
        total = sum(int(x) for x in cfg["dp1_dkvs"].split(","))
        cmd += ["--dp1-reqs", str(n), "--dp1-total-kv", str(total)]

    if args.cuda_graphs:
        cmd.append("--cuda-graphs")
    return _run_cmd(cmd, asymmetric=True)


def _run_cmd(cmd, asymmetric=False):
    """Run a torchrun command and parse BENCH_RESULT."""
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
            if asymmetric:
                return data
            avg = {}
            for c in COMPONENTS:
                avg[c] = (data["dp0"][c] + data["dp1"][c]) / 2.0
            return avg

    sys.stderr.write("  no BENCH_RESULT line found\n")
    return None


def _print_symmetric_results(groups, results):
    """Print symmetric prefill grid."""
    W = 10
    hdr = (f"  {'config':>10s}  {'reqs':>5s}  {'total_q':>7s}"
           + "".join(f"{c:>{W}s}" for c in COMPONENTS))
    sep = "  " + "\u2500" * (10 + 2 + 5 + 2 + 7 + W * len(COMPONENTS))

    for group_name, configs in groups:
        print(f"\n  {group_name}")
        print(hdr)
        print(sep)
        for cfg in configs:
            key = cfg["label"]
            if key not in results:
                continue
            avg = results[key]
            num_reqs = len(cfg["isls"])
            total_q = sum(cfg["isls"])
            cells = "".join(f"{avg[c]:>{W}.1f}" for c in COMPONENTS)
            print(f"  {key:>10s}  {num_reqs:>5d}  {total_q:>7d}{cells}")
        print(sep)


def _print_asymmetric_results(groups, results):
    """Print asymmetric prefill-vs-decode results."""
    W = 10
    hdr = (f"  {'':>50s}  {'side':>4s}"
           + "".join(f"{c:>{W}s}" for c in COMPONENTS))
    sep = "  " + "\u2500" * (50 + 2 + 4 + W * len(COMPONENTS))

    for group_name, configs in groups:
        print(f"\n  {group_name}")
        print(hdr)
        print(sep)
        for cfg in configs:
            key = cfg["label"]
            if key not in results:
                continue
            data = results[key]
            for side in ["dp0", "dp1"]:
                cells = "".join(
                    f"{data[side][c]:>{W}.1f}" for c in COMPONENTS)
                tag = "DP0" if side == "dp0" else "DP1"
                label = key if side == "dp0" else ""
                print(f"  {label:<50s}  {tag:>4s}{cells}")
            gap = abs(data["dp0"]["total"] - data["dp1"]["total"])
            slower = "DP0" if data["dp0"]["total"] > data["dp1"]["total"] \
                else "DP1"
            print(f"  {'':>50s}  gap={gap:.1f} us ({slower} slower)")
            print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Prefill grid runner for DP attention benchmark")
    parser.add_argument("--backend", default="FLASH_ATTN")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--part", choices=["dist", "imbalance", "all"],
                        default="all",
                        help="Which part to run: dist (ISL distribution), "
                             "imbalance (prefill vs decode), or all")
    args = parser.parse_args()

    run_dist = args.part in ("dist", "all")
    run_imbalance = args.part in ("imbalance", "all")

    # --- Part 1: ISL distribution ---
    sym_results = {}
    if run_dist:
        total = sum(len(cfgs) for _, cfgs in PREFILL_DIST_GROUPS)
        print(f"Part 1: ISL distribution -- {total} configs, "
              f"{args.trials} trials each\n")
        idx = 0
        for _, configs in PREFILL_DIST_GROUPS:
            for cfg in configs:
                idx += 1
                total_q = sum(cfg["isls"])
                print(f"  [{idx}/{total}] {cfg['label']} "
                      f"(total_q={total_q}) ...", end=" ", flush=True)
                avg = _run_symmetric_prefill(cfg["isls"], args)
                if avg is not None:
                    sym_results[cfg["label"]] = avg
                    print(f"attn={avg['attn']:.1f}  total={avg['total']:.1f} us")
                else:
                    print("FAILED")

    # --- Part 2: Prefill vs decode imbalance ---
    asym_results = {}
    if run_imbalance:
        total = sum(len(cfgs) for _, cfgs in IMBALANCE_CONFIGS)
        print(f"\nPart 2: Prefill vs decode imbalance -- "
              f"{total} configs\n")
        idx = 0
        for _, configs in IMBALANCE_CONFIGS:
            for cfg in configs:
                idx += 1
                print(f"  [{idx}/{total}] {cfg['label']} ...",
                      end=" ", flush=True)
                data = _run_asymmetric(cfg, args)
                if data is not None:
                    asym_results[cfg["label"]] = data
                    gap = abs(data["dp0"]["total"] - data["dp1"]["total"])
                    print(f"gap={gap:.1f} us")
                else:
                    print("FAILED")

    # --- Print summary tables ---
    print("\n" + "=" * 90)
    if sym_results:
        _print_symmetric_results(PREFILL_DIST_GROUPS, sym_results)
    if asym_results:
        _print_asymmetric_results(IMBALANCE_CONFIGS, asym_results)
    print("\n  All times in us (median, CUDA graphs ON).")
    print("=" * 90)


if __name__ == "__main__":
    main()
