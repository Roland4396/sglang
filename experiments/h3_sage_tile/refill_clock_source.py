"""Periodic-K/V diagnostic ablation; NEVER valid for arbitrary attention inputs.

Modes 1/2/3 retain the initial K/V tile and replace subsequent K/V/both TMA
transactions with zero-byte barrier arrivals. All math, tile count and phase
waits remain. The harness must repeat the quantized K and V tiles exactly.
"""
import re
from pathlib import Path
from clock_source import generate as base_generate, once


def generate(root: Path):
    src = base_generate(root)
    src = re.sub(r"probe_stamp<Profile>\(stamps, cta_stride, sample_iter, iter, \d\);", "", src)
    for name, bit, coords in [('K', 1, '0, iter * CTA_K'), ('V', 2, 'iter * CTA_K, 0')]:
        old = (f"      expect_bytes<(CTA_K * head_dim) * sizeof(int8_t)>(&barrier_{name});\n"
               f"      load_async_4D(s{name}, &tensorMap{name}, &barrier_{name}, {coords}, kv_head_id, batch_id);")
        new = (f"      if constexpr (Profile & {bit}) {{\n"
               f"        expect_bytes<0>(&barrier_{name}); // periodic-input diagnostic only\n"
               f"      }} else {{\n{old}\n      }}")
        src = once(src, old, new)
    src = once(src, "int stride, int sample_iter, bool profile)", "int stride, int sample_iter, int profile)")
    old = ("  if(profile) probe_launch<1>(q,k,v,o,qs,ks,vs,stamps,stride,sample_iter);\n"
           "  else probe_launch<0>(q,k,v,o,qs,ks,vs,stamps,stride,sample_iter);")
    new = "  switch(profile) {\n" + "\n".join(
        f"    case {i}: probe_launch<{i}>(q,k,v,o,qs,ks,vs,stamps,stride,sample_iter); break;"
        for i in range(4)) + '\n    default: TORCH_CHECK(false, "invalid diagnostic mode");\n  }'
    src = once(src, old, new)
    src = once(src, "[](bool profile){return profile ? attributes<1>() : attributes<0>();}",
               "[](int profile){return profile == 0 ? attributes<0>() : profile == 1 ? attributes<1>() : profile == 2 ? attributes<2>() : attributes<3>();}")
    return src
