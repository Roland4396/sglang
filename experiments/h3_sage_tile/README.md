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
