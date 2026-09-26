# Opt-in Comfy H3 sampling building blocks (not a complete workflow port)

Reference: ComfyUI `187eda8ef5e588c6a5765cad53e482765edae052`, T8
`6063fafbd9c3b85c5ff40aef435ae11b2844e558`.
The reviewed workflow selects **Euler + beta**, not the node's default
`dual_clock_euler + native_flow`.

New request options:
- `sigma_schedule: "comfy_beta"`: uses the 1000-point FLOW_AV video table and
  scipy Beta(.6,.6) quantiles with Comfy's rounding/deduplication. Audio follows
  the exact time-shift conversion of those video endpoints.
- `noise_layout: "comfy_av"`: the empty AV latent's video and audio noise are
  drawn sequentially with one CPU generator; audio is then packed from
  `[1,32,2,T]` to `[2*T,32]`. Original native per-modality reseeding is unchanged.
- Both default to `"native"` when absent. They are independent options, carried
  through HTTP lowering, canonical validation, resolved-plan replacement,
  and the actual timestep/latent preparation stages. The timestep dedup key
  distinguishes the schedule option.
- Existing H3 transport counts **grid points**: six Euler updates require
  `num_inference_steps=7`, with `flow_shift=12`, `audio_flow_shift=3`.

## Why the existing separate-stream Euler update is usable

Let `s = video_shift/audio_shift`, `v = sigma_video`,
`a = v/(s + (1-s)*v)` and `c = v/a = s + (1-s)*v`.
Comfy carries `y = c*x_audio` and computes derivative
`dy/dv = (1-s)*x_audio + (1+(s-1)*a)*u_audio`.
An Euler step followed by uncarrying yields
`x_next = x + (a_next-a)*u_audio` exactly in real arithmetic, because `c` is
linear in `v`. At zero sigma Comfy divides by `s` via process_latent_out.
SGLang's native velocity has the opposite sign, matching its current update.
CPU fixtures execute the pinned Comfy forward wrapper, CONST and Euler and
compare each model input and final state against SGLang's real in-place update.
Floating-point comparison is toleranced, not a claim of bit-identical inference.

## Still required before claiming author-workflow equivalence

- Exact model partition/base identity (author FL2VA checkpoint with Ref2VA task).
- Reference image/audio preprocessing, ordering, tokenizer, positions and shape.
- Attention implementation and approximation settings; do not inherit Cache-DiT
  or sparse attention silently. Correctly apply both optional author LoRAs.
- VAE precision/decoding and output encoding/audio; active first-stage scope only.
- BF16 rank-local files, actual loader hit and measured cold start.
- Authorized small real-model inference and output inspection.

These additions do not enable a new preset, switch precision, start a worker,
change the original workflow, or establish full workflow numerical equivalence.
