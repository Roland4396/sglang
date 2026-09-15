"""Regression test requiring no CUDA import/runtime: GPU weights never use a CPU file store."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

class GPUResidentVAEStoreTest(unittest.TestCase):
    def test_gpu_resident_cast_does_not_read_or_write_a_cpu_store(self):
        path=Path(__file__).resolve().parents[2]/'runtime/loader/component_loaders/vae_loader.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_rehome_cast_weights_to_file')
        ns={'torch':SimpleNamespace(dtype=object)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        vae=SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device=SimpleNamespace(type='cuda'))]))
        prepare=Mock(return_value=36)
        self.assertEqual(ns[node.name](vae,'bf16','/weights','video_vae',prepare),(36,False))
        prepare.assert_called_once_with('bf16')

if __name__=='__main__':unittest.main()
