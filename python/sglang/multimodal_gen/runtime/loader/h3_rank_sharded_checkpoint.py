from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file


FORMAT = "sglang-h3-rank-sharded-v1"
ROOT_ENV = "SGLANG_H3_RANK_SHARD_ROOT"
EXPORT_ENV = "SGLANG_H3_EXPORT_RANK_SHARDS"
STAGING_ENV = "SGLANG_H3_RANK_SHARD_STAGING_ROOT"
MARKER = ".complete.json"
COPY_BUFFER_SIZE = 16 * 1024 * 1024


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _root() -> Path | None:
    value = os.environ.get(ROOT_ENV, "").strip()
    return Path(value) if value else None


def _component_dir(component: str) -> Path | None:
    root = _root()
    return root / component if root is not None else None


def _rank_file(component_dir: Path, rank: int, world_size: int) -> Path:
    return component_dir / f"rank-{rank:05d}-of-{world_size:05d}.safetensors"


def _staging_dir(component: str) -> Path:
    configured = os.environ.get(STAGING_ENV, "").strip()
    root = (
        Path(configured)
        if configured
        else Path.home() / ".cache" / "gpu-runtime" / "h3-rank-shards"
    )
    return root / component


def source_fingerprint(source_dir: str | Path) -> str:
    source = Path(source_dir)
    digest = hashlib.sha256()
    found = False
    for name in ("config.json", "model.safetensors.index.json"):
        path = source / name
        if path.is_file():
            digest.update(name.encode("ascii"))
            digest.update(path.read_bytes())
            found = True
    if not found:
        raise RuntimeError(f"cannot fingerprint rank-shard source: {source}")
    return digest.hexdigest()


def _read_marker(component_dir: Path) -> dict | None:
    try:
        marker = json.loads((component_dir / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return marker if isinstance(marker, dict) else None


def _valid_rank_file(
    path: Path,
    component: str,
    rank: int,
    world_size: int,
    fingerprint: str,
) -> bool:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            return (
                metadata.get("format") == FORMAT
                and metadata.get("component") == component
                and metadata.get("rank") == str(rank)
                and metadata.get("world_size") == str(world_size)
                and metadata.get("source_fingerprint") == fingerprint
                and bool(list(handle.keys()))
            )
    except (OSError, ValueError):
        return False


def resolve_rank_shard(component: str, source_dir: str | Path) -> Path | None:
    component_dir = _component_dir(component)
    if component_dir is None:
        return None
    rank, world_size = _world()
    marker = _read_marker(component_dir)
    if marker is None:
        return None
    if (
        marker.get("format") != FORMAT
        or marker.get("component") != component
        or marker.get("world_size") != world_size
        or marker.get("source_fingerprint") != source_fingerprint(source_dir)
    ):
        return None
    path = _rank_file(component_dir, rank, world_size)
    fingerprint = source_fingerprint(source_dir)
    return (
        path
        if _valid_rank_file(path, component, rank, world_size, fingerprint)
        else None
    )


def _publish_rank_file(staging_file: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    with staging_file.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=COPY_BUFFER_SIZE)
        writer.flush()
    if temporary.stat().st_size != staging_file.stat().st_size:
        raise RuntimeError(f"published rank-shard size mismatch: {destination}")
    temporary.replace(destination)


def maybe_export_rank_shard(
    component: str,
    model: torch.nn.Module,
    source_dir: str | Path,
) -> None:
    if not _enabled(EXPORT_ENV):
        return
    component_dir = _component_dir(component)
    if component_dir is None:
        return
    rank, world_size = _world()
    fingerprint = source_fingerprint(source_dir)
    marker = _read_marker(component_dir)
    if (
        marker is not None
        and marker.get("format") == FORMAT
        and marker.get("component") == component
        and marker.get("world_size") == world_size
        and marker.get("source_fingerprint") == fingerprint
        and all(
            _valid_rank_file(
                _rank_file(component_dir, candidate, world_size),
                component,
                candidate,
                world_size,
                fingerprint,
            )
            for candidate in range(world_size)
        )
    ):
        return

    if rank == 0:
        component_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        (component_dir / MARKER).unlink(missing_ok=True)
    _barrier()

    destination = _rank_file(component_dir, rank, world_size)
    if not _valid_rank_file(
        destination, component, rank, world_size, fingerprint
    ):
        staging_dir = _staging_dir(component)
        staging_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging_file = _rank_file(staging_dir, rank, world_size)
        if not _valid_rank_file(
            staging_file, component, rank, world_size, fingerprint
        ):
            temporary = staging_file.with_name(f".{staging_file.name}.partial")
            temporary.unlink(missing_ok=True)
            cpu_state = {
                name: tensor.detach().to(device="cpu", copy=True).contiguous()
                for name, tensor in model.state_dict().items()
                if not tensor.is_meta
            }
            save_file(
                cpu_state,
                str(temporary),
                metadata={
                    "format": FORMAT,
                    "component": component,
                    "rank": str(rank),
                    "world_size": str(world_size),
                    "source_fingerprint": fingerprint,
                },
            )
            del cpu_state
            temporary.replace(staging_file)
        _publish_rank_file(staging_file, destination)
        if not _valid_rank_file(
            destination, component, rank, world_size, fingerprint
        ):
            raise RuntimeError(f"published rank-shard failed validation: {destination}")
        staging_file.unlink()
    _barrier()

    if rank == 0:
        missing = [
            str(_rank_file(component_dir, candidate, world_size))
            for candidate in range(world_size)
            if not _valid_rank_file(
                _rank_file(component_dir, candidate, world_size),
                component,
                candidate,
                world_size,
                fingerprint,
            )
        ]
        if missing:
            raise RuntimeError(f"rank-shard export is incomplete: {missing}")
        payload = {
            "format": FORMAT,
            "component": component,
            "world_size": world_size,
            "source_fingerprint": fingerprint,
            "created_at": time.time(),
        }
        marker_tmp = component_dir / f"{MARKER}.tmp"
        marker_tmp.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        marker_tmp.replace(component_dir / MARKER)
    _barrier()
