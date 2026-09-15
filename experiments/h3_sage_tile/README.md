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
