#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sweep direct FlashInfer paged prefill over q8 shapes that fail in vLLM."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


FAILED_SHAPES = (
    (8, 64 * 1024),
    (16, 32 * 1024),
    (16, 64 * 1024),
    (32, 32 * 1024),
    (32, 64 * 1024),
    (64, 16 * 1024),
    (64, 32 * 1024),
    (64, 64 * 1024),
)


def main() -> None:
    script = Path(__file__).with_name(
        "repro_flashinfer_prefill_noncausal_16q8s64k.py"
    )
    failures = []
    for batch_size, kv_len in FAILED_SHAPES:
        spec = f"{batch_size}q8s{kv_len // 1024}k"
        print(f"=== {spec} ===", flush=True)
        cmd = [
            sys.executable,
            str(script),
            "--batch-size",
            str(batch_size),
            "--kv-len",
            str(kv_len),
        ]
        result = subprocess.run(cmd, check=False)
        print(f"{spec}: exit={result.returncode}", flush=True)
        if result.returncode != 0:
            failures.append((spec, result.returncode))
    if failures:
        print("Failures:", failures)
        raise SystemExit(1)
    print("All direct FlashInfer shapes completed.")


if __name__ == "__main__":
    main()
