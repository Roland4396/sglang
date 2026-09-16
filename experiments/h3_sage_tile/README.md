# H3 / H20 operator experiments — not enabled in production

These are isolated CUDA experiments, **not a replacement for the installed SageAttention**.
No production backend imports them. Do not deploy merely because a numerical test passes.

## Workload and diagnosis

Native rank-zero PyTorch profiling of a synthetic H3 15-second / 50-iteration request:

- NHD Q/K/V shape `(1, 109129, 14, 128)` per rank, four H20 GPUs, TP2 × Ulysses2.
- Kernel-time sums: attention 70.53%, DeepGEMM/matrix 20.72%, communication 4.89%.
- Sampled GPU busy time 74.568 s over a 74.650 s span. CPU operator inclusive times
  are not exclusive GPU costs; do not interpret long `aten::rms_norm` CPU scopes as
  evidence that RMSNorm is the main GPU bottleneck.
- Unprofiled warm pipeline: 381.15 s; denoising 355.43 s; decoding 13.45 s.
- Profiler export and stack/shape recording overhead invalidate profiled wall time as
  a speed measurement. The profiler identifies candidates; unprofiled runs validate them.

Existing recipe remains FP8 weights, transformer Sage / shared FA, native Cache-DiT,
compile off, no LoRA, no SubBlock. The historical UI `lossless` name does not make
Cache-DiT or quantized attention numerically lossless.

## Experiments and controls

Vendored code derives from `thu-ml/SageAttention` commit
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, Apache-2.0; see `LICENSE.sageattention`.

All candidates preserve the installed 64-row Q and 128-row K per-thread INT8 scales,
FP8 V and two-level FP32 accumulation. Scope is BF16 output, head dimension 128,
NHD, noncausal, no LSE. Current CUDA stream is used explicitly.

- Q128 / one warp group: preserves independently quantized Q64 tiles; last-tile scale
  bounds are checked. Bit-identical tested output, but register spills make it slower.
- Q128 / two warp groups: share K/V, synchronize before TMA overwrite, independent
  Q64 rows. Bit-identical tested output, but much slower at long sequence length.
- Split PV columns: transient FP32 accumulators split into two 64-column operations.
  Bit-identical tested output, not a performance win without a register budget.
- K64: reuses the original K128 quantization scales. Splitting online softmax updates
  changes rounding; **not bit-identical**. Tests explicitly report both installed-Sage
  differences and differences against a BF16 FlashAttention reference.
- Register budgets: compare unconstrained / three / four resident CTA targets.
  Higher occupancy is not free: four-CTA targets can introduce local-memory spills.

Seven tested sequence lengths: 127, 128, 129, 257, 4096, 32768, 109129; small cases have
3 heads and large cases 14. Large-shape timing uses seven repetitions with rotating
implementation order. Compilation is excluded. Output checks precede timing; current
benchmark also checks every candidate on a non-default stream with explicit dependency.

At N=109129, v3 Q128 variants took 321–423 ms vs installed Sage 309 ms: **rejected**.
In v4, K128 / four-CTA target took 299.57 ms vs 303.63 ms installed; fresh-process
repeat was 299.60 vs 303.69 ms (~1.35% kernel latency reduction). This is **not a
measured end-to-end speedup**, and is not a production release. K64 variants have
~0.246% relative L2 difference from Sage at this shape, so they cannot be called exact.

v5 K128 / three-CTA target: 299.54 vs 304.59 ms (~1.66% kernel reduction), with
bit-identical output on tested shapes. This is still not an end-to-end acceptance.

## Reproduce through the managed GPU inventory

1. Query **both** eligible GPU nodes through the Ansible inventory; choose an idle GPU.
   Never infer pool availability from one host. Do not interrupt existing generations.
2. Make that node fetch/check out an exact reviewed GitHub commit. Keep experiments in
   a separate release directory; do not replace the production source symlink/package.
3. Invoke the following **node-local** command through Ansible, with a selected idle
   physical GPU and a unique output path:

```bash
CUDA_VISIBLE_DEVICES=<idle-index> \
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:/home/xql/.venvs/sglang-h3/bin:$PATH \
H3_TILE_BENCH_OUT=/home/xql/bench-result.json \
/home/xql/.venvs/sglang-h3/bin/python experiments/h3_sage_tile/bench.py
```

Use Ansible async for long tests so a controller disconnect does not kill the job.
After completion, collect the JSON/compiler log and verify the test GPU returns to zero
owned process memory. Do not stop unrelated GPU jobs.

`profile_pair.py` is a node-local synthetic warmup + native-profiler request runner.
It never reads script/novel files. Set `H3_OPERATOR_OUTPUT` for its output directory.
It uses the local orchestrator Unix socket and records exact job IDs and stage dumps.
The default native profiler exports a completed gzip trace under `/home/xql/logs`.

Summarize only the explicitly selected, completed trace (no newest-file guessing):

```bash
python3 experiments/h3_sage_tile/trace_summary.py <trace.json.gz> <summary.json>
(cd experiments/h3_sage_tile && python3 -m unittest -v test_trace_summary.py)
```

The summary separates device busy unions, cumulative kernel time and inclusive CPU
scopes. Candidate acceptance additionally needs representative video quality and paired,
unprofiled, warm **full-pipeline** measurements. Kernel numbers alone do not meet that gate.

### Software-pipeline and full-H3 canary follow-up

Tile selectors 81/82/83 add next-block QK lookahead during current-block softmax.
They preserve K128 quantization/block order and FP32+FP32 PV accumulation. The
first prologue scope bug was caught by N=129; corrected variants pass bitwise
checks but do not beat selector 71 on H20. The real-shape v8 medians were stock
308.495ms, selector71 301.327ms, lookahead81 305.654ms, lookahead82 305.751ms,
lookahead83 349.993ms. Lookahead raises register pressure (243 vs168 registers);
forcing min3 CTAs produces spills. These are microbenchmarks, not video speedups.

The opt-in backend (`SGLANG_H3_SAGE_CANARY_TILE=71`) now permits complete Comfy
A/B validation. It defaults off; unsupported shapes/causal/LSE calls use stock.
`check_canary.py` tests the entire quantization+attention wrapper, including
nondefault streams. The installed Sage V quantizers launch on legacy stream 0;
a warmed delayed-producer trace exposed cross-stream consumption of unfinished
V. `csrc/fused/v_quant.cu` contains the original Sage kernels (same upstream
license/revision as the other vendored headers) with current-stream launches.
The canary does not globally patch the installed Sage package.

Run `comfy_pair.py --label <unique-label> --output-dir <durable-directory>
--expected-tile 71` on the selected node **through the Ansible inventory**, after
checking pool/application activity and deploying a reviewed exact revision. It
uses a fixed synthetic robot prompt, 15s, 50 real steps, seed20260916, no script
reading, and records real native Comfy progress plus exact operation IDs. Retrieve
MP4s by operation ID rather than completion-order filenames. Existing production
settings must not be changed on the basis of the microbenchmark alone.


Full-video result for the opt-in selector71/current-stream-quantizer canary:
stock warm381.338s vs canary382.006s end-to-end (15s,50 real steps,TP2×Ulysses2).
Native denoising347.405s vs346.763s is not a meaningful gain. First runs462.860s
vs489.211s. Both candidate MP4s are byte-identical to their respective stock
artifacts. The candidate was therefore rolled back; production remains stock.

A further SoA shared-scratch/deferred-rescaling pipeline (selectors84/85) also
passed all bitwise checks, but took368.643/533.639ms vs306.206ms stock on the real
shape. It stays microbenchmark-only. No tested candidate achieved material video
acceleration; these experiments must not be advertised as a production speedup.

### Restricted-counter hardware diagnostics

`clock_probe.py` generates a separate CUDA extension; it never patches the installed
Sage package or starts H3. Use Ansible to select an idle GPU across the pool, expose
only that GPU, set `CUDA_HOME`, and invoke the node's H3 Python environment:

```sh
python experiments/h3_sage_tile/clock_probe.py --output-dir UNIQUE_OUTPUT
python experiments/h3_sage_tile/clock_probe.py --stamp-pair 0 7 --output-dir UNIQUE_PAIR
python experiments/h3_sage_tile/clock_probe.py --stamp-pair 3 4 --output-dir UNIQUE_SOFTMAX
python experiments/h3_sage_tile/clock_probe.py --cta-stride 1024 --output-dir UNIQUE_SPARSE
python experiments/h3_sage_tile/tensor_probe.py --output-dir UNIQUE_TENSOR
```

Only synthetic tensors are used, with captured rank shape `(1,109129,14,128)`.
The clock probe requires bitwise agreement with installed Sage at seven lengths
and on nondefault streams before collecting timings. It records installed/control/
instrumented-disabled/instrumented-active timing distributions, CUDA resource and
occupancy attributes, SASS, and sampled SM cycles. Preserve failed output directories.

The first probe inhibited contraction of output rescaling and PV accumulation:
SASS had 64 fewer FFMAs and 64 extra FMUL/FADD pairs, and N=4096 differed in
134/7340032 BF16 elements. The generator now explicitly preserves the original
fused operation in both diagnostic control and probe; do not relax the comparison
to work around an instrumentation error. Production source remains untouched.

Clock intervals contain scheduling and probe overhead, not exclusive instruction
cycles or NVIDIA stall counters. Inspect SASS marker placement and compare paired
markers/different sampling densities: a small global slowdown does not prove local
samples are unbiased. Do not convert interval percentages to Tensor Core utilization,
L2 hit rates, or attention HBM bandwidth. `tensor_probe.py` is an empirical synthetic
calibration, not a theoretical peak or a functional attention replacement. Its
launch-bounds parameter is not itself a measurement of resident blocks.

#### Follow-up: separate matrix drain and invalidate the old calibration

The original tensor calibration was invalid: it overwrote its observable result
on each iteration. SASS showed a loop decrement of four but three dummy
`HGMMA.64x8x16.F16 RZ` groups and just one intended full-matrix group. The old
~79ms alternating-QK/PV result and nominal TOPS must not be used. Each iteration
now contributes to a checked checksum; the corrected mixed median is298.224ms.
The known dummy lowering is rejected by `tensor_sass_audit.py`; static guards
do not replace inspection of loop trip counts when compiler versions change.

`clock_probe.py --matrix-split` adds pre-drain markers. Paired checks use
`--stamp-pair 2 3` (QK drain) and `--stamp-pair 7 8` (PV drain). Audit SASS:
the first clock must precede DEPBar and follow the four intended matrix issues.
These windows include probe/scheduling overhead, not exclusive hardware stalls.

`refill_probe.py` is an ablation on exactly repeating **quantized** K/V tiles,
with sequence length rounded to109184 (same CTA grid/K-iteration count as109129).
It replaces K/V/both repeated TMA transactions with zero-byte barrier arrivals,
retaining the first real tile, phase waits, and all matrix math. All four versions
match installed Sage bitwise on five periodic-input lengths/nondefault streams.
Normal301.389ms vs no-K/V-refill301.422ms: no material speedup in this control.
This is emphatically NOT a valid optimization for arbitrary/video inputs.

Together, the tests support matrix-throughput-dominated behavior for this sampled
shape on this H20, not a large avoidable repeated-transfer cost. They do not prove
an absolute whole-model lower bound, every layer's behavior, or impossibility of
better algorithms. Production was not changed and no new video speedup is claimed.
