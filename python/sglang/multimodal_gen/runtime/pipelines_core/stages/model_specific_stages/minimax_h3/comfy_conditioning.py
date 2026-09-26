# SPDX-License-Identifier: Apache-2.0
"""Explicit T8 first-stage geometry/condition contract; native remains default.

Port of T8 6063faf h3_t8/conditioning.py::_resize_reference_image (match).
This is not a claim of end-to-end model equivalence. No image/keyframe/video
hybrid or locked-source audio is admitted by this initial compatibility mode.
"""

from __future__ import annotations

import math


def normalize_conditioning_profile(value):
    if value is None:
        return "native"
    if not isinstance(value, str) or value not in {"native", "comfy_t8_match"}:
        raise ValueError("conditioning_profile must be native or comfy_t8_match")
    return value


def comfy_target_canvas(target):
    """Accept the author's actual canvas, without a second aspect/area resize."""
    dimensions = []
    for key in ("width", "height"):
        value = target.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value % 32
        ):
            raise ValueError(f"target.{key} must be a positive multiple of 32")
        dimensions.append(value)
    width, height = dimensions
    return {"width": width, "height": height}


def comfy_match_reference_shape(*, width, height, target_width, target_height):
    """No upscale, match target pixel budget, then independent nearest-32."""
    if any(
        not math.isfinite(float(v)) or float(v) <= 0
        for v in (width, height, target_width, target_height)
    ):
        raise ValueError("reference and target dimensions must be positive and finite")
    scale = min(1.0, math.sqrt((target_width * target_height) / (width * height)))
    out_w = max(32, round(width * scale / 32) * 32)
    out_h = max(32, round(height * scale / 32) * 32)
    return {
        "geometry": "reference_image_resolved",
        "shape_policy_version": "comfy_t8_match_v1",
        "width": out_w,
        "height": out_h,
        "multiple": 32,
        "rounding": "nearest",
        "allow_upscale": False,
    }


def reference_audio_duration_limit(profile, material_chain, target_duration):
    # T8 encodes the complete standalone reference, not only target seconds.
    if profile == "comfy_t8_match" and material_chain == "audio":
        return None
    return target_duration


def comfy_qwen_image_inputs(images):
    """Comfy 187eda8 qwen_vl image preprocessing, with H3's patch16/mean.5.

    Qwen and the VAE must see the same already-resized reference. Avoid HF
    processor defaults (including interpolation/pixel budgets) changing it.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    values, grids = [], []
    for image in images:
        x = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).float() / 255.0
        height, width = x.shape[:2]
        h, w = round(height / 32) * 32, round(width / 32) * 32
        if h * w > 12845056:
            beta = math.sqrt(height * width / 12845056)
            h, w = (
                max(32, math.floor(height / beta / 32) * 32),
                max(32, math.floor(width / beta / 32) * 32),
            )
        elif h * w < 3136:
            beta = math.sqrt(3136 / (height * width))
            h, w = math.ceil(height * beta / 32) * 32, math.ceil(width * beta / 32) * 32
        x = F.interpolate(
            x.permute(2, 0, 1).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0]
        x = (x - 0.5) / 0.5
        gh, gw = h // 16, w // 16
        patches = (
            x.unsqueeze(0)
            .repeat(2, 1, 1, 1)
            .reshape(1, 2, 3, gh // 2, 2, 16, gw // 2, 2, 16)
            .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshape(gh * gw, 3 * 2 * 16 * 16)
        )
        values.append(patches)
        grids.append([1, gh, gw])
    return {
        "pixel_values": torch.cat(values),
        "image_grid_thw": torch.tensor(grids, dtype=torch.long),
    }
