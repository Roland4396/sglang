from pathlib import Path
import re
import unittest
from matrix_clock_source import generate


class MatrixClockTests(unittest.TestCase):
    def test_ten_ordered_markers_and_buffer_size(self):
        s = generate(Path(__file__).resolve().parent)
        indices = re.findall(r"probe_stamp<Profile>\(stamps, cta_stride, sample_iter, iter, (\d)\);", s)
        self.assertEqual(indices, [str(x) for x in range(10)])
        self.assertIn("(linear / cta_stride) * 10 + phase", s)
        self.assertIn("q.size(0),stride)*10);", s)
        for i in (2, 7):
            self.assertIn(f"sample_iter, iter, {i});\n    wgmma::warpgroup_wait<0>();", s)

    def test_wait_pair(self):
        s = generate(Path(__file__).resolve().parent, (2, 3))
        self.assertEqual(s.count("probe_stamp<Profile>("), 2)
        with self.assertRaises(ValueError): generate(Path(__file__).resolve().parent, (3, 2))

    def test_refill_ablation_preserves_waits_and_has_no_clocks(self):
        from refill_clock_source import generate as refill
        s = refill(Path(__file__).resolve().parent)
        self.assertNotIn("probe_stamp<Profile>(", s)
        self.assertEqual(s.count("expect_bytes<0>(&barrier_K)"), 1)
        self.assertEqual(s.count("expect_bytes<0>(&barrier_V)"), 1)
        self.assertIn("wait(&barrier_K, p);", s)
        self.assertIn("wait(&barrier_V, p);", s)
        for i in range(4): self.assertIn(f"case {i}: probe_launch<{i}>", s)


if __name__ == "__main__": unittest.main()
