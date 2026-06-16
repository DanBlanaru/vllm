#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Direct FlashInfer FI-prefill-vs-XQA q8 comparison.

This script does not use vLLM internals. It compares:

* FlashInfer native paged prefill
* FlashInfer TRT-LLM/XQA decode API with ``backend="xqa"`` and q_len_per_req=8

The default shape is a representative q8 speculative-verification proxy cell
where the benchmark showed XQA faster than FI:

  16 requests, q_len=8, kv_len=16k, 32 Q heads, 8 KV heads, head_dim=128.

Run from a vLLM checkout or any environment with torch + flashinfer installed:

  python benchmarks/attention_benchmarks/compare_flashinfer_prefill_vs_xqa.py
"""

from __future__ import annotations

import argparse
import math

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.decode import trtllm_batch_decode_with_kv_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--q-len", type=int, default=8)
    parser.add_argument("--kv-len", type=int, default=16 * 1024)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--q-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--kv-dtype", choices=("fp8_e4m3", "float16", "bfloat16"), default="fp8_e4m3")
    parser.add_argument("--fi-causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workspace-mib", type=int, default=394)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def kv_torch_dtype(name: str) -> torch.dtype:
    return {
        "fp8_e4m3": torch.float8_e4m3fn,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def make_indptr(count: int, stride: int, device: str = "cpu") -> torch.Tensor:
    return torch.arange(
        0,
        (count + 1) * stride,
        stride,
        dtype=torch.int32,
        device=device,
    )


def make_xqa_spec_decode_causal_mask(
    batch_size: int,
    q_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return XQA bit-packed causal mask for uniform q_len speculative decode."""
    num_packed_masks_per_token = (q_len + 31) // 32
    q_indices = torch.arange(q_len, device=device, dtype=torch.int32).unsqueeze(1)
    kv_indices = torch.arange(q_len, device=device, dtype=torch.int32).unsqueeze(0)
    causal_bool_mask = kv_indices <= q_indices

    padded_seq_len = num_packed_masks_per_token * 32
    if padded_seq_len > q_len:
        padding = torch.zeros(
            q_len,
            padded_seq_len - q_len,
            device=device,
            dtype=torch.bool,
        )
        causal_bool_mask = torch.cat([causal_bool_mask, padding], dim=1)

    causal_bool_mask = causal_bool_mask.view(q_len, num_packed_masks_per_token, 32)
    bit_positions = torch.tensor(
        [1 << i for i in range(32)],
        device=device,
        dtype=torch.int64,
    )
    mask_uint32 = (
        (causal_bool_mask.to(torch.int64) * bit_positions).sum(dim=-1).to(torch.uint32)
    )
    mask_uint32 = (
        mask_uint32.unsqueeze(0)
        .expand(batch_size, q_len, num_packed_masks_per_token)
        .contiguous()
    )
    return mask_uint32.view(torch.uint16)


def make_kv_cache(
    total_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    layout: str,
    source_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> torch.Tensor:
    if layout == "NHD":
        shape = (total_pages, 2, page_size, num_kv_heads, head_dim)
    else:
        shape = (total_pages, 2, num_kv_heads, page_size, head_dim)
    return torch.randn(*shape, dtype=source_dtype, device="cuda").to(kv_dtype)


def time_cuda(fn, warmup_iters: int, iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    args = parse_args()
    if args.kv_len % args.page_size != 0:
        raise ValueError("This script expects kv_len to be page-aligned.")
    if args.q_len <= 1:
        raise ValueError("This comparison is intended for q_len > 1.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_default_device("cuda")

    q_dtype = torch_dtype(args.q_dtype)
    kv_dtype = kv_torch_dtype(args.kv_dtype)
    pages_per_req = args.kv_len // args.page_size
    total_pages = args.batch_size * pages_per_req
    total_q = args.batch_size * args.q_len
    sm_scale = 1.0 / math.sqrt(args.head_dim)

    print(
        "shape:",
        {
            "batch_size": args.batch_size,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "page_size": args.page_size,
            "num_qo_heads": args.num_qo_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "kv_layout": args.kv_layout,
            "q_dtype": str(q_dtype),
            "kv_dtype": str(kv_dtype),
        },
    )

    q = torch.randn(
        total_q,
        args.num_qo_heads,
        args.head_dim,
        dtype=q_dtype,
        device="cuda",
    )
    kv_cache = make_kv_cache(
        total_pages,
        args.page_size,
        args.num_kv_heads,
        args.head_dim,
        args.kv_layout,
        q_dtype,
        kv_dtype,
    )
    fi_out = torch.empty_like(q)
    xqa_out = torch.empty_like(q)

    block_tables = torch.arange(
        total_pages,
        dtype=torch.int32,
        device="cuda",
    ).view(args.batch_size, pages_per_req)
    seq_lens = torch.full(
        (args.batch_size,),
        args.kv_len,
        dtype=torch.int32,
        device="cuda",
    )

    workspace = torch.zeros(
        args.workspace_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )

    prefill = BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        args.kv_layout,
        backend="auto",
    )
    prefill.plan(
        qo_indptr=make_indptr(args.batch_size, args.q_len),
        paged_kv_indptr=make_indptr(args.batch_size, pages_per_req),
        paged_kv_indices=torch.arange(total_pages, dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len=torch.full(
            (args.batch_size,),
            args.page_size,
            dtype=torch.int32,
            device="cpu",
        ),
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim_qk=args.head_dim,
        page_size=args.page_size,
        causal=args.fi_causal,
        sm_scale=sm_scale,
        window_left=-1,
        logits_soft_cap=0.0,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
        o_data_type=q_dtype,
        fixed_split_size=-1,
        disable_split_kv=False,
    )

    xqa_mask = make_xqa_spec_decode_causal_mask(
        args.batch_size,
        args.q_len,
        q.device,
    )

    def run_fi() -> None:
        prefill.run(
            q,
            kv_cache,
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            out=fi_out,
            kv_cache_sf=None,
        )

    def run_xqa() -> None:
        trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=args.kv_len,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            window_left=-1,
            out=xqa_out,
            kv_layout=args.kv_layout,
            q_len_per_req=args.q_len,
            backend="xqa",
            mask=xqa_mask,
            kv_cache_sf=None,
        )

    fi_ms = time_cuda(run_fi, args.warmup_iters, args.iters)
    xqa_ms = time_cuda(run_xqa, args.warmup_iters, args.iters)

    fi_label = "fi_prefill_causal" if args.fi_causal else "fi_prefill_noncausal"
    print(f"{fi_label}_ms={fi_ms:.3f}")
    print(f"xqa_decode_causal_ms={xqa_ms:.3f}")
    print(f"xqa_vs_fi={fi_ms / xqa_ms:.3f}x")
    print("fi_checksum", fi_out.float().sum().item())
    print("xqa_checksum", xqa_out.float().sum().item())


if __name__ == "__main__":
    main()
