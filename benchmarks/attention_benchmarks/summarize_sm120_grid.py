#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize SM120 q8 grid artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BENCH_ROOT = Path("/home/scratch/scratch.dblanaru/bench_serving")
TRITON_XQA_JSON = (
    BENCH_ROOT
    / "artifacts/sm120_specdec_kernel_microbench_grid_triton_xqa_nocg.json"
)
FI_DIR = BENCH_ROOT / "artifacts/sm120_fi_noncausal_one_by_one"
FI_SUMMARY = FI_DIR / "summary.csv"
OUT_MD = BENCH_ROOT / "artifacts/sm120_specdec_kernel_microbench_grid_final.md"
OUT_CSV = BENCH_ROOT / "artifacts/sm120_specdec_kernel_microbench_grid_final.csv"
OUT_FI_MD = BENCH_ROOT / "artifacts/sm120_specdec_kernel_microbench_grid_fi_indexed.md"
OUT_FI_CSV = BENCH_ROOT / "artifacts/sm120_specdec_kernel_microbench_grid_fi_indexed.csv"


def load_results(path: Path) -> dict[tuple[str, str], dict]:
    with path.open() as f:
        rows = json.load(f)
    return {
        (row["config"]["batch_spec"], row["config"]["backend"]): row
        for row in rows
        if row.get("error") is None
    }


def load_fi_status() -> dict[str, str]:
    if not FI_SUMMARY.exists():
        return {}
    with FI_SUMMARY.open() as f:
        return {row["batch_spec"]: row["status"] for row in csv.DictReader(f)}


def load_fi_rows() -> dict[str, dict]:
    rows = {}
    if not FI_SUMMARY.exists():
        return rows
    with FI_SUMMARY.open() as f:
        for summary_row in csv.DictReader(f):
            if summary_row["status"] != "OK":
                continue
            json_path = Path(summary_row["json"])
            if not json_path.exists():
                continue
            with json_path.open() as jf:
                for result in json.load(jf):
                    if result["config"]["backend"] == "fi_prefill_noncausal":
                        rows[summary_row["batch_spec"]] = result
                        break
    return rows


def fmt_time(row: dict | None) -> str:
    if row is None:
        return "ERR"
    return f"{row['mean_time'] * 1e6:.1f}us"


def fmt_speed(row: dict | None, triton: dict | None) -> str:
    if row is None or triton is None:
        return "ERR"
    return f"{triton['mean_time'] / row['mean_time']:.2f}x"


def fmt_diff(row: dict | None) -> str:
    if row is None:
        return "ERR"
    val = row.get("output_max_abs_diff_vs_triton")
    if val is None:
        return "n/a"
    return f"{val:.3e}"


def fmt_backend_vs_fi(row: dict | None, fi: dict | None) -> str:
    if row is None or fi is None:
        return "ERR"
    return f"{fi['mean_time'] / row['mean_time']:.2f}x"


def main() -> None:
    tx = load_results(TRITON_XQA_JSON)
    fi_status = load_fi_status()
    fi_rows_by_spec = load_fi_rows()
    specs = sorted(
        {spec for spec, _ in tx} | set(fi_status),
        key=lambda spec: (
            int(spec.split("q", 1)[0] or 1),
            int(spec.rsplit("s", 1)[1].removesuffix("k")),
        ),
    )

    rows = []
    for spec in specs:
        triton = tx.get((spec, "triton"))
        xqa = tx.get((spec, "xqa_decode_causal"))
        fi = fi_rows_by_spec.get(spec)
        fi_status_str = fi_status.get(spec, "MISSING")
        if fi_status_str != "OK":
            fi = None
        rows.append(
            {
                "batch_spec": spec,
                "triton_time": fmt_time(triton),
                "xqa_speed_vs_triton": fmt_speed(xqa, triton),
                "fi_speed_vs_triton": fmt_speed(fi, triton)
                if fi_status_str == "OK"
                else "ERR",
                "xqa_max_abs_diff": fmt_diff(xqa),
                "fi_max_abs_diff": fmt_diff(fi) if fi_status_str == "OK" else "ERR",
                "fi_status": fi_status_str,
            }
        )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n")

    fi_rows = []
    for spec in specs:
        triton = tx.get((spec, "triton"))
        xqa = tx.get((spec, "xqa_decode_causal"))
        fi = fi_rows_by_spec.get(spec)
        fi_status_str = fi_status.get(spec, "MISSING")
        if fi_status_str != "OK":
            fi = None
        fi_rows.append(
            {
                "batch_spec": spec,
                "fi_time": fmt_time(fi) if fi is not None else "ERR",
                "fi": "1.00x" if fi is not None else "ERR",
                "xqa_vs_fi": fmt_backend_vs_fi(xqa, fi),
                "triton_vs_fi": fmt_backend_vs_fi(triton, fi),
                "xqa_max_abs_diff": fmt_diff(xqa),
                "fi_max_abs_diff": fmt_diff(fi) if fi is not None else "ERR",
                "fi_status": fi_status_str,
            }
        )

    with OUT_FI_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fi_rows[0]))
        writer.writeheader()
        writer.writerows(fi_rows)

    fi_headers = list(fi_rows[0])
    fi_lines = [
        "| " + " | ".join(fi_headers) + " |",
        "| " + " | ".join(["---"] * len(fi_headers)) + " |",
    ]
    for row in fi_rows:
        fi_lines.append("| " + " | ".join(row[h] for h in fi_headers) + " |")
    OUT_FI_MD.write_text("\n".join(fi_lines) + "\n")
    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_FI_MD)
    print(OUT_FI_CSV)
    print("\n".join(lines))
    print()
    print("\n".join(fi_lines))


if __name__ == "__main__":
    main()
