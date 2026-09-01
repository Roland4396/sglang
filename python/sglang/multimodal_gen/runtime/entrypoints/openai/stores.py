import asyncio
from typing import Any, Dict, Optional


class AsyncDictStore:
    """A small async-safe in-memory key-value store for dict items.

    This encapsulates the usual pattern of a module-level dict guarded by
    an asyncio.Lock and provides simple CRUD methods that are safe to call
    concurrently from FastAPI request handlers and background tasks.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, key: str, value: Dict[str, Any]) -> None:
        async with self._lock:
            self._items[key] = value

    async def insert_if_absent(
        self, key: str, value: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Insert once and return the existing item when the key is reserved.

        This is used by long-running media creates so a replayed HTTP request
        cannot start a second generation while the first request is still
        preparing its scheduler batch.
        """

        async with self._lock:
            existing = self._items.get(key)
            if existing is not None:
                return existing
            self._items[key] = value
            return None

    async def update_fields(
        self, key: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            item.update(updates)
            return item

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._items.get(key)

    async def pop(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._items.pop(key, None)

    async def list_page(
        self,
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = "desc",
    ) -> list[Dict[str, Any]]:
        """Return a created-time ordered page, with OpenAI-style cursor semantics."""
        normalized_order = (order or "desc").lower()
        reverse = normalized_order != "asc"
        async with self._lock:
            items = sorted(
                self._items.values(),
                key=lambda item: item.get("created_at", 0),
                reverse=reverse,
            )

        if after is not None:
            try:
                index = next(i for i, item in enumerate(items) if item["id"] == after)
                items = items[index + 1 :]
            except StopIteration:
                return []

        return items[:limit] if limit is not None else items


# Global stores shared by OpenAI entrypoints
# [request_id, dict]
VIDEO_STORE = AsyncDictStore()
IMAGE_STORE = AsyncDictStore()
MESH_STORE = AsyncDictStore()
