import unittest
from trace_summary import interval_union, kernel_kind, summarize


class TraceSummaryTests(unittest.TestCase):
    def test_union_not_sum_for_overlap(self):
        self.assertEqual(interval_union([(50, 150), (0, 100), (160, 170)]), 160)
        self.assertEqual(interval_union([]), 0)

    def test_attention_precedes_matrix_substring(self):
        self.assertEqual(kernel_kind('attention_wgmma'), 'attention')
        self.assertEqual(kernel_kind('ncclAllReduce'), 'communication')
        self.assertEqual(kernel_kind('deep_gemm'), 'matrix')

    def test_devices_are_not_combined(self):
        events = [dict(cat='kernel', name='deep_gemm', ts=0, dur=100, args={'device': 0}),
                  dict(cat='kernel', name='attention', ts=50, dur=100, args={'device': 0}),
                  dict(cat='kernel', name='attention', ts=50, dur=100, args={'device': 1}),
                  dict(cat='cpu_op', name='aten::rms_norm', ts=0, dur=300)]
        result = summarize(events)
        self.assertEqual(result['kernel_sum_us'], 300)
        self.assertEqual(result['devices']['0']['busy_union_us'], 150)
        self.assertEqual(result['devices']['1']['busy_union_us'], 100)
        self.assertEqual(result['groups_sum_us']['attention'], 200)
        self.assertEqual(result['cpu_ops_inclusive'][0]['total_us'], 300)

    def test_empty_trace(self):
        self.assertEqual(summarize([])['kernel_sum_us'], 0)


if __name__ == '__main__':
    unittest.main()
