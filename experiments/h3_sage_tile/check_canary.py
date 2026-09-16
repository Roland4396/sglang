"""Real GPU check of the complete quantization+kernel bridge, before H3 tests."""
import importlib.util
import json
import os
from pathlib import Path

import torch
from sageattention import sageattn

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "canary", root / "python/sglang/multimodal_gen/runtime/layers/attention/backends/h3_sage_canary.py")
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)
tile = int(os.environ.get("SGLANG_H3_SAGE_CANARY_TILE", "71"))

with torch.inference_mode():
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
                                  finite=finite, bitwise_equal=equal)), flush=True)
            assert finite and equal
        del q, k, v, reference, result
print("PASS complete canary wrapper", flush=True)
