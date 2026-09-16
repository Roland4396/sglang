"""CPU-only tests: opt-in dispatch must never broaden the tested kernel scope."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "python/sglang/multimodal_gen/runtime/layers/attention/backends/h3_sage_canary.py"


class Tensor:
    ndim = 4
    is_cuda = True
    device = "cuda:0"
    dtype = "bf16"
    shape = (1, 129, 14, 128)

    def stride(self, dim):
        return 1


class GateTests(unittest.TestCase):
    def setUp(self):
        self.torch = types.SimpleNamespace(
            bfloat16="bf16", cuda=types.SimpleNamespace(get_device_capability=lambda _: (9, 0)))
        with patch.dict(sys.modules, {"torch": self.torch}):
            spec = importlib.util.spec_from_file_location("canary_test", FILE)
            self.mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.mod)
        self.q, self.k, self.v = Tensor(), Tensor(), Tensor()

    def test_supported_h20(self):
        self.assertTrue(self.mod.supports(self.q, self.k, self.v))

    def test_lse_and_causal_stay_stock(self):
        self.assertFalse(self.mod.supports(self.q, self.k, self.v, return_lse=True))
        self.assertFalse(self.mod.supports(self.q, self.k, self.v, is_causal=True))

    def test_other_hardware_stays_stock(self):
        self.torch.cuda.get_device_capability = lambda _: (8, 0)
        self.assertFalse(self.mod.supports(self.q, self.k, self.v))

    def test_shape_dtype_device_and_strides(self):
        for attr, value in (("shape", (1, 128, 14, 128)), ("dtype", "fp16"),
                            ("device", "cuda:1"), ("ndim", 3), ("is_cuda", False)):
            v = Tensor()
            setattr(v, attr, value)
            self.assertFalse(self.mod.supports(self.q, self.k, v), attr)
        self.v.stride = lambda _: 2
        self.assertFalse(self.mod.supports(self.q, self.k, self.v))

    def test_empty_stays_stock(self):
        for t in (self.q, self.k, self.v):
            t.shape = (1, 0, 14, 128)
        self.assertFalse(self.mod.supports(self.q, self.k, self.v))

    def test_source_path(self):
        self.assertTrue((FILE.parents[7] / "experiments/h3_sage_tile/csrc/qattn/h3_tile.cu").is_file())


if __name__ == "__main__":
    unittest.main()
