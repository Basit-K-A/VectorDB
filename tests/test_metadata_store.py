"""Tests for in-memory metadata storage."""

import pytest

from vectordb.metadata.base import MetadataRecord
from vectordb.metadata.metadata_store import InMemoryMetadataStore


@pytest.fixture
def store() -> InMemoryMetadataStore:
    return InMemoryMetadataStore()


@pytest.fixture
def sample_records() -> list[MetadataRecord]:
    return [
        MetadataRecord(
            id="doc-1",
            collection="articles",
            data={"category": "finance", "author": "john", "year": 2025},
        ),
        MetadataRecord(
            id="doc-2",
            collection="articles",
            data={"category": "finance", "author": "jane", "year": 2024},
        ),
        MetadataRecord(
            id="doc-3",
            collection="articles",
            data={"category": "tech", "author": "john", "year": 2023},
        ),
    ]


class TestInMemoryMetadataStore:
    def test_upsert_and_get(
        self,
        store: InMemoryMetadataStore,
        sample_records: list[MetadataRecord],
    ) -> None:
        store.upsert_sync(sample_records)
        records = store.get_sync("articles", ["doc-1", "doc-3"])

        assert len(records) == 2
        assert records[0].data["author"] == "john"

    def test_delete(
        self,
        store: InMemoryMetadataStore,
        sample_records: list[MetadataRecord],
    ) -> None:
        store.upsert_sync(sample_records)
        deleted = store.delete_sync("articles", ["doc-2"])

        assert deleted == 1
        assert store.filter_sync("articles", {"author": "jane"}) == []

    def test_filter_by_equality(
        self,
        store: InMemoryMetadataStore,
        sample_records: list[MetadataRecord],
    ) -> None:
        store.upsert_sync(sample_records)
        ids = store.filter_sync("articles", {"category": "finance"})
        assert ids == ["doc-1", "doc-2"]

    def test_filter_by_range(
        self,
        store: InMemoryMetadataStore,
        sample_records: list[MetadataRecord],
    ) -> None:
        store.upsert_sync(sample_records)
        ids = store.filter_sync(
            "articles",
            {"year": {"$gte": 2024, "$lte": 2025}},
        )
        assert ids == ["doc-1", "doc-2"]

    def test_filter_by_multi_condition(
        self,
        store: InMemoryMetadataStore,
        sample_records: list[MetadataRecord],
    ) -> None:
        store.upsert_sync(sample_records)
        ids = store.filter_sync(
            "articles",
            {
                "$and": [
                    {"category": "finance"},
                    {"author": "john"},
                ]
            },
        )
        assert ids == ["doc-1"]

    def test_list_sync(
        self,
        store: InMemoryMetadataStore,
        sample_records: list[MetadataRecord],
    ) -> None:
        store.upsert_sync(sample_records)
        assert len(store.list_sync("articles")) == 3

    def test_get_metadata(
        self,
        store: InMemoryMetadataStore,
        sample_records: list[MetadataRecord],
    ) -> None:
        store.upsert_sync(sample_records)
        metadata = store.get_metadata("articles", "doc-1")
        assert metadata == {"category": "finance", "author": "john", "year": 2025}

    async def test_async_interface(
        self,
        store: InMemoryMetadataStore,
        sample_records: list[MetadataRecord],
    ) -> None:
        await store.upsert(sample_records)
        records = await store.get("articles", ["doc-1"])
        ids = await store.filter("articles", {"author": "john"})
        deleted = await store.delete("articles", ["doc-3"])

        assert len(records) == 1
        assert "doc-1" in ids
        assert deleted == 1
