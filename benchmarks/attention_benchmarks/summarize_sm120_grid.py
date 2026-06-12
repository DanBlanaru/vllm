#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize SM120 q8 grid artifacts."""

from __future__ import annotations

import csv
import json
import math
from argparse import ArgumentParser, Namespace
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


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Summarize SM120 q8 Triton/XQA grid results and FI one-by-one "
            "results. Defaults reproduce the original handoff tables."
        )
    )
    parser.add_argument(
        "--triton-xqa-json",
        action="append",
        type=Path,
        dest="triton_xqa_jsons",
        help=(
            "Triton/XQA benchmark JSON. May be repeated; later files override "
            "duplicate (batch_spec, backend) rows."
        ),
    )
    parser.add_argument(
        "--fi-summary",
        type=Path,
        default=FI_SUMMARY,
        help="FI one-by-one summary.csv with per-shape JSON paths.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=OUT_MD,
        help="Output Markdown path for the Triton-indexed table.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=OUT_CSV,
        help="Output CSV path for the Triton-indexed table.",
    )
    parser.add_argument(
        "--out-fi-md",
        type=Path,
        default=OUT_FI_MD,
        help="Output Markdown path for the FI-indexed table.",
    )
    parser.add_argument(
        "--out-fi-csv",
        type=Path,
        default=OUT_FI_CSV,
        help="Output CSV path for the FI-indexed table.",
    )
    return parser.parse_args()


def load_results(paths: list[Path]) -> dict[tuple[str, str], dict]:
    results = {}
    for path in paths:
        with path.open() as f:
            for row in json.load(f):
                if is_valid_result(row):
                    config = row["config"]
                    results[(config["batch_spec"], config["backend"])] = row
    return results


def is_valid_result(row: dict) -> bool:
    return row.get("error") is None and math.isfinite(row.get("mean_time", math.inf))


def status_from_summary_row(summary_row: dict) -> str:
    status = summary_row["status"]
    exit_code = summary_row.get("exit_code", "")
    if status != "OK" and exit_code and exit_code != "0":
        return f"EXIT_{exit_code}"
    return status


def status_from_invalid_result(result: dict | None) -> str:
    if result is None:
        return "NO_FI_ROW"
    if not math.isfinite(result.get("mean_time", math.inf)):
        return "BENCH_INF"
    if result.get("error") is not None:
        return "BENCH_ERR"
    return "ERR"


def load_fi_data(fi_summary: Path) -> tuple[dict[str, str], dict[str, dict]]:
    statuses = {}
    rows = {}
    if not fi_summary.exists():
        return statuses, rows
    with fi_summary.open() as f:
        for summary_row in csv.DictReader(f):
            spec = summary_row["batch_spec"]
            statuses[spec] = status_from_summary_row(summary_row)
            if statuses[spec] != "OK":
                continue
            json_path = Path(summary_row["json"])
            if not json_path.exists():
                statuses[spec] = "MISSING_JSON"
                continue
            fi_result = None
            with json_path.open() as jf:
                for result in json.load(jf):
                    if result["config"]["backend"] == "fi_prefill_noncausal":
                        fi_result = result
                        break
            if is_valid_result(fi_result):
                rows[spec] = fi_result
            else:
                statuses[spec] = status_from_invalid_result(fi_result)
    return statuses, rows


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


def write_table(rows: list[dict], path: Path) -> None:
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spec_sort_key(spec: str) -> tuple[int, int, int]:
    batch_str, rest = spec.split("q", 1)
    q_len_str, seq_len_str = rest.split("s", 1)
    seq_len = int(seq_len_str.removesuffix("k"))
    if seq_len_str.endswith("k"):
        seq_len *= 1024
    return (int(batch_str or 1), int(q_len_str), seq_len)


def main() -> None:
    args = parse_args()
    triton_xqa_jsons = args.triton_xqa_jsons or [TRITON_XQA_JSON]

    tx = load_results(triton_xqa_jsons)
    fi_status, fi_rows_by_spec = load_fi_data(args.fi_summary)
    specs = sorted(
        {spec for spec, _ in tx} | set(fi_status),
        key=spec_sort_key,
    )

    rows = []
    for spec in specs:
        triton = tx.get((spec, "triton"))
        xqa = tx.get((spec, "xqa_decode_causal"))
        fi = fi_rows_by_spec.get(spec)
        fi_status_str = fi_status.get(spec, "MISSING")
        if fi_status_str == "OK" and fi is None:
            fi_status_str = "ERR"
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

    write_csv(rows, args.out_csv)
    write_table(rows, args.out_md)

    fi_rows = []
    for spec in specs:
        triton = tx.get((spec, "triton"))
        xqa = tx.get((spec, "xqa_decode_causal"))
        fi = fi_rows_by_spec.get(spec)
        fi_status_str = fi_status.get(spec, "MISSING")
        if fi_status_str == "OK" and fi is None:
            fi_status_str = "ERR"
        if fi_status_str != "OK":
            fi = None
        fi_rows.append(
            {
                "batch_spec": spec,
                "fi_time": fmt_time(fi) if fi is not None else "ERR",
                "xqa_vs_fi": fmt_backend_vs_fi(xqa, fi),
                "triton_vs_fi": fmt_backend_vs_fi(triton, fi),
                "fi_max_abs_diff": fmt_diff(fi) if fi is not None else "ERR",
                "xqa_max_abs_diff": fmt_diff(xqa),
                "fi_status": fi_status_str,
            }
        )

    write_csv(fi_rows, args.out_fi_csv)
    write_table(fi_rows, args.out_fi_md)

    print(args.out_md)
    print(args.out_csv)
    print(args.out_fi_md)
    print(args.out_fi_csv)


if __name__ == "__main__":
    main()
