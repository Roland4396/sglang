"""Real GPU check of the complete quantization+kernel bridge, before H3 tests."""
import importlib.util
import json
import os
from pathlib import Path

import torch
from sageattention import sageattn
from sageattention.core import per_channel_fp8

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "canary", root / "python/sglang/multimodal_gen/runtime/layers/attention/backends/h3_sage_canary.py")
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)
tile = int(os.environ.get("SGLANG_H3_SAGE_CANARY_TILE", "71"))

with torch.inference_mode():
    # A deterministic stream-order probe: a delayed producer fills V with ones.
    # A quantizer using legacy stream 0 can race this producer. Record old and
    # candidate separately; only the candidate must satisfy the ordering gate.
    canary.load_kernel()
    for name in ("installed", "canary"):
        v = torch.zeros(1, 4096, 14, 128, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        # Warm this exact stream's allocator: a fresh cudaMalloc can hide the
        # race by synchronizing the device before the legacy-stream launches.
        with torch.cuda.stream(stream):
            warm = (per_channel_fp8(v, tensor_layout="NHD", smooth_v=False)
                    if name == "installed" else canary.quantize_v(v))
        torch.cuda.synchronize()
        del warm
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
            v.fill_(1)
            if name == "installed":
                vi, vs, _ = per_channel_fp8(v, tensor_layout="NHD", smooth_v=False)
            else:
                vi, vs = canary.quantize_v(v)
        torch.cuda.synchronize()
        correct = bool((vi.float() == 448).all()) and bool(torch.allclose(vs, torch.full_like(vs, 1 / 448)))
        print(json.dumps(dict(probe="delayed_v_producer", implementation=name,
                              ordered=correct, v_scale_max=vs.max().item())), flush=True)
        if name == "canary":
            assert correct, "V quantizer violated caller stream ordering"
        del v, vi, vs
    for n in (127, 128, 129, 257, 4096, 109129):
        torch.manual_seed(20260916 + n)
        q, k, v = [torch.randn(1, n, 14, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        k += 2.0  # Exercise key centering, not only zero-mean random inputs.
        assert canary.supports(q, k, v)
        reference = sageattn(q, k, v, tensor_layout="NHD", sm_scale=128 ** -0.5)
        for nondefault in (False, True):
            stream = torch.cuda.Stream() if nondefault else torch.cuda.current_stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                result = canary.forward(q, k, v, sm_scale=128 ** -0.5, tile=tile)
            stream.synchronize()
            finite = bool(torch.isfinite(result).all())
            equal = torch.equal(reference, result)
            print(json.dumps(dict(n=n, tile=tile, nondefault_stream=nondefault,
                                  finite=finite, bitwise_equal=equal,
                                  max_abs=(reference.float()-result.float()).abs().max().item())), flush=True)
            assert finite and equal
        del q, k, v, reference, result
print("PASS complete canary wrapper", flush=True)
