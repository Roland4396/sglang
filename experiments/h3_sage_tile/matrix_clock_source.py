"""Split WGMMA issue/commit from final drain; diagnostic source only.

The pre-wait timestamp must be verified in SASS to precede DEPBar. A compiler
inserted drain before the timestamp invalidates the split, even if values match.
"""
from pathlib import Path
import re

from clock_source import generate as base_generate, once

PHASES = ["k_scale_and_ready", "qk_issue_and_commit", "qk_drain_wait",
          "k_prefetch_issue", "softmax_denominator_and_fp8", "v_ready_wait",
          "pv_issue_and_commit", "pv_drain_wait",
          "output_rescale_accumulate_and_v_prefetch"]


def generate(root: Path, stamp_pair=None):
    if stamp_pair is not None and not (len(stamp_pair) == 2 and 0 <= stamp_pair[0] < stamp_pair[1] < 10):
        raise ValueError("Expected two ordered timestamp indices from 0 through 9")
    src = base_generate(root)
    mapping = [0, 1, 3, 4, 5, 6, 8, 9]
    pattern = r"probe_stamp<Profile>\(stamps, cta_stride, sample_iter, iter, (\d)\);"
    src = re.sub(pattern, lambda m: m[0].replace(", " + m[1] + ");", ", " + str(mapping[int(m[1])]) + ");"), src)
    prefix, rest = src.split("  int p = 1;", 1)
    loop, tail = rest.split("\n\n  { \n    p ^= 1;", 1)
    anchor = "    wgmma::warpgroup_commit_batch();\n    wgmma::warpgroup_wait<0>();"
    assert loop.count(anchor) == 2, "Expected QK and unsplit PV drain points"
    for index in (2, 7):
        replacement = ("    wgmma::warpgroup_commit_batch();\n"
                       f"    probe_stamp<Profile>(stamps, cta_stride, sample_iter, iter, {index});\n"
                       "    wgmma::warpgroup_wait<0>();")
        loop = loop.replace(anchor, replacement, 1)
    src = prefix + "  int p = 1;" + loop + "\n\n  { \n    p ^= 1;" + tail
    src = once(src, "(linear / cta_stride) * 8 + phase", "(linear / cta_stride) * 10 + phase")
    src = once(src, "q.size(0),stride)*8);", "q.size(0),stride)*10);")
    if stamp_pair is not None:
        src = re.sub(pattern, lambda m: m[0] if int(m[1]) in stamp_pair else "", src)
    assert src.count("probe_stamp<Profile>(") == (10 if stamp_pair is None else 2)
    return src
