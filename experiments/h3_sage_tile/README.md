# H3 long-sequence Sage tile experiment (not enabled in production)

Trace on H20 identifies the SM90 Sage attention kernel as ~71% of sampled kernel time.
This experiment tests a 128-query CTA while preserving 64-row per-thread quantization,
128-key tiles, FP8 V, and two-level FP32 PV accumulation. Padding-scale bounds and
CUDA current-stream execution are explicit. A 64-query compiled control is mandatory.
No sparse pattern, LoRA, weight or model settings are changed.

Vendored headers/kernel derive from thu-ml/SageAttention commit
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, Apache-2.0; see LICENSE.sageattention.
Modifications: per-query-tile Q scale indexing, specialized host interface, and current
CUDA stream launch. Validation must compare outputs and timing before any deployment.

The first 128-query / single-warp-group prototype was bit-identical in all tested
shapes but slower (327 vs 308 ms at N=109129) because ptxas reported 255 registers
and 504/512 bytes of spill stores/loads. Variant 256 instead assigns two independent
64-query tiles to two warp groups in one CTA sharing K/V. CTA barriers precede K/V
TMA overwrite; per-thread quantization and accumulation order remain unchanged.
