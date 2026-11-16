import asyncio
import copy
import json

import pytest

from ingestion.pilot import embed_chunks
from pipelines.budget import ServiceError


def test_completed_embedding_batches_survive_later_failure_without_repeat_spending(tmp_path):
    chunks = [{"id": str(index), "text": f"source {index}"} for index in range(41)]
    path = tmp_path / "chunks.json"

    class FailingModels:
        def __init__(self):
            self.calls = 0

        async def embed(self, texts):
            self.calls += 1
            if self.calls == 3:
                raise ServiceError("third batch failed")
            return [[0.1, 0.2] for _ in texts]

    first = FailingModels()
    with pytest.raises(ServiceError):
        asyncio.run(embed_chunks(first, copy.deepcopy(chunks), path))
    cached = json.loads(path.read_text())
    assert sum("vector" in chunk for chunk in cached["chunks"]) == 32

    class ResumedModels:
        def __init__(self):
            self.requests = []

        async def embed(self, texts):
            self.requests.append(texts)
            return [[0.1, 0.2] for _ in texts]

    second = ResumedModels()
    complete = asyncio.run(embed_chunks(second, copy.deepcopy(chunks), path))
    assert second.requests == [[f"source {index}" for index in range(32, 41)]]
    assert all("vector" in chunk for chunk in complete)
    asyncio.run(embed_chunks(second, copy.deepcopy(chunks), path))
    assert len(second.requests) == 1


def test_cache_mismatch_does_not_spend_model_calls(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps({"source_fingerprint": "mismatch", "chunks": []}))

    class NeverCall:
        async def embed(self, texts):
            pytest.fail("stale source cache must fail before inference")

    with pytest.raises(ValueError, match="altered"):
        asyncio.run(embed_chunks(NeverCall(), [{"id": "1", "text": "source"}], path))


def test_source_updates_reembed_only_changed_records_when_explicitly_enabled(tmp_path):
    class Models:
        def __init__(self):
            self.requests = []

        async def embed(self, texts):
            self.requests.append(texts)
            return [[0.1, 0.2] for _ in texts]

    models = Models()
    path = tmp_path / "chunks.json"
    original = [{"id": "one", "text": "old"}, {"id": "two", "text": "unchanged"}]
    asyncio.run(embed_chunks(models, copy.deepcopy(original), path))
    changed = [{"id": "one", "text": "new"}, original[1]]
    with pytest.raises(ValueError, match="do not match"):
        asyncio.run(embed_chunks(models, copy.deepcopy(changed), path))
    updated = asyncio.run(
        embed_chunks(models, copy.deepcopy(changed), path, allow_source_update=True)
    )
    assert models.requests == [["old", "unchanged"], ["new"]]
    assert len(updated) == 2 and all("vector" in item for item in updated)


def test_reordered_unchanged_sources_persist_without_any_new_model_calls(tmp_path):
    class Models:
        def __init__(self):
            self.calls = 0

        async def embed(self, texts):
            self.calls += 1
            return [[0.1, 0.2] for _ in texts]

    models = Models()
    path = tmp_path / "chunks.json"
    original = [{"id": "one", "text": "first"}, {"id": "two", "text": "second"}]
    asyncio.run(embed_chunks(models, copy.deepcopy(original), path))
    asyncio.run(embed_chunks(models, copy.deepcopy(original[::-1]), path, allow_source_update=True))
    saved = json.loads(path.read_text())
    assert [item["id"] for item in saved["chunks"]] == ["two", "one"]
    assert models.calls == 1
    asyncio.run(embed_chunks(models, copy.deepcopy(original[::-1]), path))
    assert models.calls == 1
