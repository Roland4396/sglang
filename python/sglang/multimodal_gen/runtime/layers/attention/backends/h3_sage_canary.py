"""Opt-in H20 kernel canary; not enabled by any production profile.

The quantization and two-level FP32 accumulation match SageAttention SM90.
Kernel build/execution errors deliberately propagate instead of disguising a
failed canary as a successful stock run.
"""

import functools
import os
from pathlib import Path

import torch


def supports(q, k, v, *, is_causal=False, return_lse=False):
    return (
        not is_causal
        and not return_lse
        and all(t.ndim == 4 and t.is_cuda for t in (q, k, v))
        and q.device == k.device == v.device
        and all(t.dtype == torch.bfloat16 and t.shape[-1] == 128
                and t.stride(-1) == 1 for t in (q, k, v))
        and q.shape == k.shape == v.shape
        and q.shape[1] > 0
        and torch.cuda.get_device_capability(q.device) == (9, 0)
    )


@functools.lru_cache(maxsize=1)
def load_kernel():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[7] / "experiments/h3_sage_tile"
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    return load(
        name="h3_sage_tile_20260916_v6",
        sources=[str(root / "csrc/qattn/h3_tile.cu")],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "--ptxas-options=-v",
        ],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )


def forward(q, k, v, *, sm_scale, tile):
    from sageattention.core import per_channel_fp8, per_thread_int8_triton

    with torch.cuda.device(q.device):
        kernel = load_kernel()
        km = k.mean(dim=1, keepdim=True)
        qi, qs, ki, ks = per_thread_int8_triton(
            q, k, km, tensor_layout="NHD", BLKQ=64, WARPQ=16,
            BLKK=128, WARPK=128,
        )
        pad = (-k.shape[1]) % 128
        if pad:
            v = torch.cat([v, v.new_zeros(v.shape[0], pad, v.shape[2], 128)], dim=1)
        vi, vs, _ = per_channel_fp8(v, tensor_layout="NHD", smooth_v=False)
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        kernel.forward(qi, ki, vi, out, qs, ks, vs,
                       128 ** -0.5 if sm_scale is None else sm_scale, tile)
        return out
