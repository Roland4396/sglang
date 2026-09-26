# SPDX-License-Identifier: Apache-2.0
"""Comfy RandomNoise order for an empty joint AV latent, batch size one."""

from __future__ import annotations


def normalize_noise_layout(value: str | None) -> str:
    if value is None:
        return "native"
    if not isinstance(value, str) or value not in ("native", "comfy_av"):
        raise ValueError("noise_layout must be 'native' or 'comfy_av'")
    return value


def comfy_av_initial_noise(*, seed: int, video_shape: tuple[int, ...], audio_t: int):
    """Draw video then [B,C,stereo,T] audio from ONE CPU generator.

    The normal H3 recipe independently re-seeds audio and draws row-major noise.
    Neither is the Comfy NestedTensor RandomNoise recipe. Keep this opt-in;
    changing the existing recipe would silently change users' seeded outputs.
    Returned video is in raw latent order; audio is channel-major stereo rows.
    This helper does not cover masks, latent restarts, or nonzero input latents.
    """
    import torch

    if len(video_shape) != 5 or video_shape[:2] != (1, 24):
        raise ValueError("comfy_av video_shape must be [1, 24, T, H, W]")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in video_shape):
        raise ValueError("comfy_av video dimensions must be positive integers")
    if isinstance(audio_t, bool) or not isinstance(audio_t, int) or audio_t < 1:
        raise ValueError("comfy_av audio_t must be a positive integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video = torch.randn(
        video_shape, generator=generator, device="cpu", dtype=torch.float32
    )
    audio = torch.randn(
        (1, 32, 2, audio_t), generator=generator, device="cpu", dtype=torch.float32
    )
    audio_rows = audio.permute(0, 2, 3, 1).reshape(2 * audio_t, 32).contiguous()
    return video, audio_rows
