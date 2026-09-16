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

    def test_ambiguous_source_rejected(self):
        with self.assertRaises(ValueError): once("aaa", "a", "b")


if __name__ == "__main__": unittest.main()
