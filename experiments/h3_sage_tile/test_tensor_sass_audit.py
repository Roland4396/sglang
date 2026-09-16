import unittest
from tensor_sass_audit import audit_tensor_sass


class TensorAuditTests(unittest.TestCase):
    def test_known_bad_lowering_rejected(self):
        text = 'Function : tensor_probe_kernel\n HGMMA.64x8x16.F16 RZ, gdesc[URZ], RZ;'
        with self.assertRaisesRegex(ValueError, 'dummy WGMMA'): audit_tensor_sass(text)

    def test_missing_kernels_rejected(self):
        with self.assertRaises(ValueError): audit_tensor_sass('')

    def test_full_instruction_fixture(self):
        s = ''.join('Function : tensor_probe_kernel_'+str(i)+'\n' +
                    'IGMMA.64x128x32.S8.S8 R24;\n'*4 for i in range(6))
        self.assertTrue(audit_tensor_sass(s)['known_dummy_lowering_absent'])


if __name__ == '__main__': unittest.main()
