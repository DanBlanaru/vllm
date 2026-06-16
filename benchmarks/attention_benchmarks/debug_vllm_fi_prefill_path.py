#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Debug vLLM FI native prefill vs direct FlashInfer wrapper call."""

from __future__ import annotations

import os

import torch

from common import BenchmarkConfig
from runner import (
    _build_common_attn_metadata,
    _create_backend_impl,
    _create_input_tensors,
    _create_kv_cache,
    _create_kv_cache_spec,
    _create_metadata_builder,
    _create_vllm_config,
    _get_backend_config,
    benchmark_flashinfer_env,
)
from vllm.config import set_current_vllm_config
from vllm.v1.attention.backends.flashinfer import FIPrefill, FlashInferBackend
from vllm.v1.attention.backends.utils import get_kv_cache_layout, set_kv_cache_layout


def main() -> None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config = BenchmarkConfig(
        backend="fi_prefill_noncausal",
        attention_kernel="fi_prefill_noncausal",
        batch_spec="16q8s64k",
        num_layers=10,
        head_dim=128,
        num_q_heads=32,
        num_kv_heads=8,
        block_size=16,
        device="cuda:0",
        repeats=1,
        warmup_iters=1,
        kv_cache_dtype="fp8",
        use_cuda_graphs=False,
    )
    device = torch.device(config.device)
    torch.accelerator.set_device_index(device)

    q_lens = [8] * 16
    kv_lens = [64 * 1024] * 16
    total_q = sum(q_lens)
    max_blocks_per_request = (max(kv_lens) + config.block_size - 1) // config.block_size
    max_num_blocks = len(q_lens) * max_blocks_per_request

    with benchmark_flashinfer_env(config):
        vllm_config = _create_vllm_config(config, max_num_blocks)
        dtype = vllm_config.model_config.dtype
        with set_current_vllm_config(vllm_config):
            backend_cfg = _get_backend_config("FLASHINFER")
            backend_class, impl, layer = _create_backend_impl(
                backend_cfg, config, device, dtype
            )
            required_layout = backend_class.get_required_kv_cache_layout()
            set_kv_cache_layout(required_layout)
            get_kv_cache_layout.cache_clear()
            print("layout", get_kv_cache_layout())

            common_metadata = _build_common_attn_metadata(
                q_lens, kv_lens, config.block_size, device
            )
            kv_cache_spec = _create_kv_cache_spec(config, dtype)
            builder = _create_metadata_builder(
                backend_class, kv_cache_spec, vllm_config, device, "FLASHINFER"
            )
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_metadata,
            )
            assert isinstance(attn_metadata.prefill, FIPrefill)
            wrapper = attn_metadata.prefill.wrapper
            print("q_data_type_prefill", attn_metadata.q_data_type_prefill)
            print("impl.kv_cache_dtype", impl.kv_cache_dtype)
            print("num_prefills/tokens", attn_metadata.num_prefills, attn_metadata.num_prefill_tokens)
            print("wrapper", type(wrapper))
            print("wrapper index attrs", [a for a in dir(wrapper) if "indices" in a])

            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            q_list, k_list, v_list = _create_input_tensors(
                config, total_q, device, dtype, quantize_query=False
            )
            cache_list = _create_kv_cache(
                config, max_num_blocks, backend_class, device, dtype
            )
            out = torch.empty(
                total_q, config.num_q_heads, config.head_dim, device=device, dtype=dtype
            )
            stride_order = FlashInferBackend.get_kv_cache_stride_order()
            kv_cache_permute = cache_list[0].permute(*stride_order)
            print("kv logical", cache_list[0].shape, cache_list[0].stride())
            print("kv permute", kv_cache_permute.shape, kv_cache_permute.stride())
            indices = getattr(wrapper, "_paged_kv_indices_buf", None)
            if indices is not None:
                expected = torch.arange(indices.numel(), dtype=indices.dtype, device=indices.device)
                print("indices shape/stride", indices.shape, indices.stride())
                print("indices min/max", indices.min().item(), indices.max().item())
                print("indices arange equal", torch.equal(indices, expected))

            print("direct wrapper.run with vLLM-built objects")
            wrapper.run(
                q_list[0],
                kv_cache_permute,
                q_scale=layer._q_scale_float,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                out=out,
                kv_cache_sf=None,
            )
            torch.cuda.synchronize()
            print("direct wrapper.run OK", out.float().sum().item())

            print("impl.forward")
            impl.forward(
                layer,
                q_list[0],
                k_list[0],
                v_list[0],
                cache_list[0],
                attn_metadata,
                output=out,
            )
            torch.cuda.synchronize()
            print("impl.forward OK", out.float().sum().item())


if __name__ == "__main__":
    main()
