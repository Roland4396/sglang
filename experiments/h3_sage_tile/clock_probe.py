"""Hardware-clock diagnostic only. Run via Ansible on one verified idle H20.

Uses synthetic tensors at the captured H3 rank shape. Does not start H3, change
production, read prompts, or claim exclusive stall counters from elapsed cycles.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from clock_source import generate, PHASES

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--repeats", type=int, default=9)
args = parser.parse_args()
root = Path(__file__).resolve().parent
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
assert not (out / "result.json").exists(), "Choose a new output directory"
source = generate(root)
(out / "clock_probe.generated.cu").write_text(source)

import torch
from torch.utils.cpp_extension import load
from sageattention.core import per_thread_int8_triton, per_channel_fp8, sm90_compile
from sageattention import _qattn_sm90

os.environ.setdefault("MAX_JOBS", "2")
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
assert torch.cuda.device_count() == 1, "Expose only the selected idle GPU"
assert torch.cuda.get_device_capability() == (9, 0)
start = time.time()
module = load(name="h3_clock_probe_20260916_v1", sources=[str(out / "clock_probe.generated.cu")],
              extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "--ptxas-options=-v",
                  "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                  "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"],
              extra_ldflags=["-lcuda"], verbose=True)
results = {"build_seconds": time.time() - start, "phases": PHASES,
           "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
           "installed_binary": _qattn_sm90.__file__, "probe_binary": module.__file__,
           "attributes_order": ["registers_per_thread", "local_bytes_per_thread", "static_shared_bytes", "resident_ctas_per_sm"],
           "attributes": {str(p): module.attributes(p) for p in (False, True)},
           "scope": "sampled CTA/thread-0 elapsed clocks, not exclusive instruction cycles or performance-event counters",
           "checks": [], "timings": {}, "samples": []}
for label, path in [("installed", _qattn_sm90.__file__), ("probe", module.__file__)]:
    results[label + "_binary_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for flag, suffix in [("-res-usage", "resources"), ("-sass", "sass")]:
        with (out / (label + "-" + suffix + ".txt")).open("w") as f:
            subprocess.run([os.environ["CUDA_HOME"] + "/bin/cuobjdump", flag, path], stdout=f, stderr=subprocess.STDOUT, check=True)
print("ATTRIBUTES", json.dumps(results["attributes"]), flush=True)


def save():
    (out / "result.json").write_text(json.dumps(results, indent=2))


@torch.inference_mode()
def prepare(n, heads):
    torch.manual_seed(20260916 + n)
    q, k, v = [torch.randn(1, n, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    qi, qs, ki, ks = per_thread_int8_triton(q, k, k.mean(dim=1, keepdim=True),
        tensor_layout="NHD", BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)
    if n % 128:
        v = torch.cat([v, v.new_zeros(1, (-n) % 128, heads, 128)], dim=1)
    vi, vs, _ = per_channel_fp8(v, tensor_layout="NHD", smooth_v=False)
    torch.cuda.synchronize()  # quantization is outside timed/probed attention
    tensors = qi, ki, vi, qs, ks, vs
    buffers = {name: torch.empty_like(q) for name in ("installed", "control", "probe_disabled", "probe_active")}
    stamps = torch.zeros(math.ceil(math.ceil(n / 64) * heads / 128), 8, device="cuda", dtype=torch.int64)
    return tensors, buffers, stamps


def funcs(tensors, buffers, stamps, iteration):
    qi, ki, vi, qs, ks, vs = tensors
    return {
        "installed": lambda: sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(qi, ki, vi, buffers["installed"], qs, ks, vs, 0, 0, 3, 128**-0.5, 0),
        "control": lambda: module.run(qi, ki, vi, buffers["control"], qs, ks, vs, stamps, 128, iteration, False),
        "probe_disabled": lambda: module.run(qi, ki, vi, buffers["probe_disabled"], qs, ks, vs, stamps, 128, -1, True),
        "probe_active": lambda: module.run(qi, ki, vi, buffers["probe_active"], qs, ks, vs, stamps, 128, iteration, True),
    }


with torch.inference_mode():
    for n in (127, 128, 129, 257, 4096, 32768, 109129):
        heads = 3 if n < 4096 else 14
        tensors, buffers, stamps = prepare(n, heads)
        fns = funcs(tensors, buffers, stamps, 1)
        for fn in fns.values(): fn()
        torch.cuda.synchronize()
        for name, value in buffers.items():
            assert torch.isfinite(value).all().item(), (n, name, "nonfinite")
            if not torch.equal(value, buffers["installed"]):
                def delta(a, b):
                    af, bf = a.float(), b.float()
                    d = af - bf
                    return {"different": (a != b).sum().item(), "elements": a.numel(),
                            "max_abs": d.abs().max().item(),
                            "relative_l2": (d.norm() / bf.norm()).item()}
                failure = {"n": n, "name": name, "variants": {}, "repeat_stable": {}}
                for variant, tensor in buffers.items():
                    failure["variants"][variant] = delta(tensor, buffers["installed"])
                    previous = tensor.clone()
                    fns[variant](); torch.cuda.synchronize()
                    failure["repeat_stable"][variant] = delta(tensor, previous)
                results["numerical_failure"] = failure
                save()
                print("NUMERICAL_FAILURE", json.dumps(failure), flush=True)
            assert torch.equal(value, buffers["installed"]), (n, name, "numerical mismatch")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for name in ("control", "probe_disabled", "probe_active"): fns[name]()
        stream.synchronize()
        assert all(torch.equal(x, buffers["installed"]) for x in buffers.values())
        results["checks"].append({"n": n, "heads": heads, "bitwise_equal": True, "finite": True, "nondefault_stream": True})
        print("CHECK", n, "PASS", flush=True)
        save()
        if n != 109129:
            del tensors, buffers, stamps, fns
            torch.cuda.empty_cache()

    props = torch.cuda.get_device_properties(0)
    results["device"] = str(props)
    fns = funcs(tensors, buffers, stamps, 426)
    for _ in range(3):
        for fn in fns.values(): fn()
    torch.cuda.synchronize()
    times = {name: [] for name in fns}
    for repeat in range(args.repeats):
        names = list(fns)
        names = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in names:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); fns[name](); b.record(); b.synchronize()
            times[name].append(a.elapsed_time(b))
    results["timings"] = {"samples_ms": times, "median_ms": {k: statistics.median(v) for k, v in times.items()}}
    med = results["timings"]["median_ms"]
    results["perturbation_vs_control"] = {k: med[k] / med["control"] - 1 for k in ("probe_disabled", "probe_active")}
    print("TIMINGS", json.dumps(results["timings"]), flush=True)
    print("PERTURBATION", json.dumps(results["perturbation_vs_control"]), flush=True)
    for iteration in (1, 64, 128, 256, 426, 640, 800, 852):
        for repeat in range(2):
            stamps.zero_()
            funcs(tensors, buffers, stamps, iteration)["probe_active"]()
            clocks = stamps.cpu()
            assert (clocks > 0).all().item(), "Missing stamps"
            differences = clocks[:, 1:] - clocks[:, :-1]
            assert (differences > 0).all().item(), "Nonmonotonic clock samples"
            row = {"iteration": iteration, "repeat": repeat, "cta_stride": 128,
                   "elapsed_cycles": differences.tolist(),
                   "median_cycles": differences.double().median(dim=0).values.tolist()}
            results["samples"].append(row)
            print("SAMPLE", iteration, repeat, row["median_cycles"], flush=True)
    results["sample_mean_cycles"] = torch.tensor([x for row in results["samples"] for x in row["elapsed_cycles"]], dtype=torch.float64).mean(dim=0).tolist()
    results["interpretation_gate"] = "tentative only: sparse sampling can perturb selected CTAs even if global timing stays similar; inspect SASS and paired control before attributing phases"
    save()
    print("PASS", json.dumps({k: results[k] for k in ("attributes", "perturbation_vs_control", "sample_mean_cycles")}), flush=True)
