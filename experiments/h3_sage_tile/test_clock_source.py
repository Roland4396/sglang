from pathlib import Path
import unittest
from clock_source import generate, once


class ClockSourceTests(unittest.TestCase):
    def test_probe_stays_in_main_loop(self):
        s = generate(Path(__file__).resolve().parent)
        loop = s.split("  int p = 1;", 1)[1].split("\n  }\n\n  { \n    p ^= 1;", 1)[0]
        self.assertEqual(loop.count("probe_stamp<Profile>("), 8)
        for phase in range(8):
            self.assertIn(f"sample_iter, iter, {phase});", loop)
        self.assertIn("int Profile=0>", s)
        self.assertIn("cudaOccupancyMaxActiveBlocksPerMultiprocessor", s)
        self.assertNotIn("m.def(\"forward\"", s)
        self.assertEqual(loop.count("__fmaf_rn(RO[fq][fv][k]"), 1)
        self.assertIn("tile_sm_scale, deferred_scale);", loop)

    def test_ambiguous_source_rejected(self):
        with self.assertRaises(ValueError): once("aaa", "a", "b")

    def test_pair_only_has_two_clock_calls(self):
        s = generate(Path(__file__).resolve().parent, (0, 7))
        self.assertEqual(s.count("probe_stamp<Profile>("), 2)
        for pair in [(7, 0), (0, 8), (3, 3)]:
            with self.assertRaises(ValueError): generate(Path(__file__).resolve().parent, pair)


if __name__ == "__main__": unittest.main()
