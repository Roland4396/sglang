"""Small, job-scoped CPU telemetry for an external video control plane.

The denoiser calls this on rank zero after completed steps. No tensor copies,
CUDA synchronization, network requests, or shared-filesystem writes are needed.
"""

import json
import logging
import os
import re
import time
from pathlib import Path


def write_video_progress(request_id: str, step: int, total: int, phase: str) -> None:
    root = os.environ.get("SGLANG_VIDEO_PROGRESS_DIR", "")
    if not root or not re.fullmatch(r"[0-9a-fA-F-]{36}", str(request_id)):
        return
    path = Path(root) / f"{request_id}.json"
    staging = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps({
            "job_id": request_id, "step": int(step), "total": int(total),
            "phase": phase, "updated_at": time.time(),
        }), encoding="utf-8")
        staging.replace(path)
    except OSError as exc:
        # Telemetry must not turn an otherwise successful generation into a failure.
        logging.getLogger(__name__).warning("video_progress_write_failed: %s", exc)
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
