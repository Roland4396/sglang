from __future__ import annotations

import asyncio
import importlib.util
import unittest
from pathlib import Path


STORE_PATH = (
    Path(__file__).resolve().parents[2]
    / "runtime"
    / "entrypoints"
    / "openai"
    / "stores.py"
)
SPEC = importlib.util.spec_from_file_location("h3_video_stores_test", STORE_PATH)
assert SPEC is not None and SPEC.loader is not None
STORE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STORE_MODULE)
AsyncDictStore = STORE_MODULE.AsyncDictStore


class VideoIdempotencyStoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_insert_if_absent_has_one_winner(self) -> None:
        store = AsyncDictStore()
        first = {"id": "same", "status": "queued", "producer": "first"}
        second = {"id": "same", "status": "queued", "producer": "second"}

        results = await asyncio.gather(
            store.insert_if_absent("same", first),
            store.insert_if_absent("same", second),
        )

        self.assertEqual(sum(result is None for result in results), 1)
        stored = await store.get("same")
        self.assertIn(stored, (first, second))
        loser_result = next(result for result in results if result is not None)
        self.assertIs(loser_result, stored)


if __name__ == "__main__":
    unittest.main()
