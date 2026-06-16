#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Direct FlashInfer paged prefill repro for q8 non-causal 16x64k.

This bypasses vLLM's attention backend and calls FlashInfer's
BatchPrefillWithPagedKVCacheWrapper directly. It mirrors the failing
benchmark shape:

  - 16 requests
  - q_len = 8
  - kv_len = 64k
  - page_size = 16
  - NHD paged KV cache layout
  - fp8 KV cache
  - bf16 query/output
  - causal = False

Run from a vLLM checkout with FlashInfer installed:

  python benchmarks/attention_benchmarks/repro_flashinfer_prefill_noncausal_16q8s64k.py
"""

from __future__ import annotations

import argparse
import math

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--q-len", type=int, default=8)
    parser.add_argument("--kv-len", type=int, default=64 * 1024)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--kv-layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--q-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace-mib", type=int, default=394)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument(
        "--vllm-logical-cache-view",
        action="store_true",
        help=(
            "Allocate physical HND cache, then expose it as vLLM's logical "
            "NHD-shaped view and permute back before calling FlashInfer."
        ),
    )
    parser.add_argument(
        "--vllm-copy-page-indices",
        action="store_true",
        help="Build paged_kv_indices via vLLM's Triton copy kernel.",
    )
    parser.add_argument(
        "--allocate-unused-kv-inputs",
        action="store_true",
        help="Allocate unused K/V input tensors like the vLLM benchmark runner.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def make_indptr(count: int, stride: int) -> torch.Tensor:
    return torch.arange(
        0,
        (count + 1) * stride,
        stride,
        dtype=torch.int32,
        device="cpu",
    )


def main() -> None:
    args = parse_args()
    if args.kv_len % args.page_size != 0:
        raise ValueError("This compact repro expects kv_len to be page-aligned.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_default_device("cuda")

    pages_per_req = args.kv_len // args.page_size
    total_pages = args.batch_size * pages_per_req
    total_q = args.batch_size * args.q_len
    sm_scale = 1.0 / math.sqrt(args.head_dim)
    q_dtype = torch_dtype(args.q_dtype)

    print(
        "shape:",
        {
            "batch_size": args.batch_size,
            "q_len": args.q_len,
            "kv_len": args.kv_len,
            "page_size": args.page_size,
            "total_pages": total_pages,
            "num_qo_heads": args.num_qo_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "kv_layout": args.kv_layout,
            "q_dtype": q_dtype,
        },
    )

    workspace = torch.zeros(
        args.workspace_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        args.kv_layout,
        backend="auto",
    )

    qo_indptr = make_indptr(args.batch_size, args.q_len)
    paged_kv_indptr = make_indptr(args.batch_size, pages_per_req)
    if args.vllm_copy_page_indices:
        from vllm.v1.attention.backends.flashinfer import _copy_page_indices_kernel

        block_table = torch.arange(
            total_pages,
            dtype=torch.int32,
            device="cuda",
        ).view(args.batch_size, pages_per_req)
        paged_kv_indices = torch.empty(
            total_pages,
            dtype=torch.int32,
            device="cuda",
        )
        paged_kv_indptr_gpu = paged_kv_indptr.to("cuda")
        _copy_page_indices_kernel[(args.batch_size,)](
            paged_kv_indices,
            block_table,
            block_table.stride(0),
            paged_kv_indptr_gpu,
            BLOCK_SIZE=1024,
        )
    else:
        paged_kv_indices = torch.arange(
            total_pages,
            dtype=torch.int32,
            device="cuda",
        )
    paged_kv_last_page_len = torch.full(
        (args.batch_size,),
        args.page_size,
        dtype=torch.int32,
        device="cpu",
    )

    q_list = [
        torch.randn(
            total_q,
            args.num_qo_heads,
            args.head_dim,
            dtype=q_dtype,
            device="cuda",
        )
        for _ in range(args.num_layers)
    ]
    if args.allocate_unused_kv_inputs:
        _k_list = [
            torch.randn(
                total_q,
                args.num_kv_heads,
                args.head_dim,
                dtype=q_dtype,
                device="cuda",
            )
            for _ in range(args.num_layers)
        ]
        _v_list = [
            torch.randn(
                total_q,
                args.num_kv_heads,
                args.head_dim,
                dtype=q_dtype,
                device="cuda",
            )
            for _ in range(args.num_layers)
        ]
    kv_cache_list = []
    for _ in range(args.num_layers):
        physical_shape = (
            (total_pages, 2, args.page_size, args.num_kv_heads, args.head_dim)
            if args.kv_layout == "NHD"
            else (total_pages, 2, args.num_kv_heads, args.page_size, args.head_dim)
        )
        physical_hnd = torch.randn(
            *physical_shape,
            dtype=q_dtype,
            device="cuda",
        ).to(torch.float8_e4m3fn)
        if args.vllm_logical_cache_view and args.kv_layout == "HND":
            logical_view = physical_hnd.permute(0, 1, 3, 2, 4)
            kv_cache = logical_view.permute(0, 1, 3, 2, 4)
        elif args.vllm_logical_cache_view:
            kv_cache = physical_hnd.permute(0, 1, 2, 3, 4)
        else:
            kv_cache = physical_hnd
        kv_cache_list.append(kv_cache)
    print("kv_cache shape/stride:", kv_cache_list[0].shape, kv_cache_list[0].stride())
    out = torch.empty_like(q_list[0])

    print("planning")
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim_qk=args.head_dim,
        page_size=args.page_size,
        causal=False,
        sm_scale=sm_scale,
        window_left=-1,
        logits_soft_cap=0.0,
        q_data_type=q_dtype,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=q_dtype,
        fixed_split_size=-1,
        disable_split_kv=False,
    )

    print("running")
    for _ in range(args.warmup_iters):
        for q, kv_cache in zip(q_list, kv_cache_list):
            wrapper.run(
                q,
                kv_cache,
                q_scale=1.0,
                k_scale=1.0,
                v_scale=1.0,
                out=out,
                kv_cache_sf=None,
            )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        for q, kv_cache in zip(q_list, kv_cache_list):
            wrapper.run(
                q,
                kv_cache,
                q_scale=1.0,
                k_scale=1.0,
                v_scale=1.0,
                out=out,
                kv_cache_sf=None,
            )
    end.record()
    torch.cuda.synchronize()
    print(f"elapsed_ms_per_layer={(start.elapsed_time(end) / args.iters / args.num_layers):.3f}")
    out_cpu = out.detach().float().cpu()
    print("out_checksum", out_cpu.sum().item())


if __name__ == "__main__":
    main()
