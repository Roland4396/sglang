# SPDX-License-Identifier: Apache-2.0
"""Opt-in Comfy FLOW_AV beta schedule; no Comfy runtime dependency.

Reference: ComfyUI 187eda8ef5e588c6a5765cad53e482765edae052,
ModelSamplingAV / ModelSamplingDiscreteFlow and beta_scheduler(alpha=beta=.6).
This is a schedule option, NOT a claim of full workflow equivalence.
"""

from __future__ import annotations

import math


def normalize_sigma_schedule(value: str | None) -> str:
    if value is None:
        return "native"
    if not isinstance(value, str) or value not in ("native", "comfy_beta"):
        raise ValueError("sigma_schedule must be 'native' or 'comfy_beta'")
    return value


def comfy_beta_av_sigmas(
    *, num_steps: int, video_shift: float, audio_shift: float
) -> dict[str, list[float]]:
    """Return N+1 endpoints for N Euler updates (duplicates follow Comfy).

    ``num_steps`` means actual updates here. The H3 transport historically
    counts grid points, so its caller passes ``num_inference_steps - 1``.
    Audio is derived from the selected video sigmas, not independently sampled.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("comfy_beta requires at least one Euler update")
    for name, value in (("video_shift", video_shift), ("audio_shift", audio_shift)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a positive finite number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")

    import numpy as np
    import scipy.stats
    import torch

    # Preserve Comfy's fp32 operations including multiply/divide by 1000.
    timestep = (torch.arange(1, 1001, device="cpu", dtype=torch.float32) / 1000) * 1000
    base = timestep / 1000
    table = video_shift * base / (1 + (video_shift - 1) * base)
    percentiles = 1 - np.linspace(0, 1, num_steps, endpoint=False)
    indices = np.rint(scipy.stats.beta.ppf(percentiles, 0.6, 0.6) * 999)
    selected = []
    previous = -1
    for index in indices:
        index = int(index)
        if index != previous:
            selected.append(table[index])
        previous = index
    video = torch.tensor([*selected, 0.0], dtype=torch.float32, device="cpu")
    # Same rational clock mapping as Comfy's MiniMaxH3Model.time_shift_sigma.
    base = video / (video_shift + video * (1.0 - video_shift))
    audio = audio_shift * base / (1.0 + (audio_shift - 1.0) * base)
    return {"video": video.tolist(), "audio": audio.tolist()}
