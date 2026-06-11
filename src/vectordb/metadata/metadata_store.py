"""In-memory metadata storage with filter support."""

from collections import defaultdict

from vectordb.metadata.base import MetadataRecord, MetadataStore
from vectordb.metadata.filters import matches_metadata
from vectordb.types import CollectionName, Metadata, MetadataFilter, VectorId


class InMemoryMetadataStore(MetadataStore):
    """Store metadata records in memory and filter them by collection."""

    def __init__(self) -> None:
        self._records: dict[CollectionName, dict[VectorId, MetadataRecord]] = defaultdict(dict)

    def upsert_sync(self, records: list[MetadataRecord]) -> None:
        """Insert or update metadata records synchronously."""
        for record in records:
            self._records[record.collection][record.id] = record

    def get_sync(
        self,
        collection: CollectionName,
        ids: list[VectorId],
    ) -> list[MetadataRecord]:
        """Retrieve metadata records synchronously."""
        collection_records = self._records.get(collection, {})
        return [
            collection_records[record_id]
            for record_id in ids
            if record_id in collection_records
        ]

    def delete_sync(
        self,
        collection: CollectionName,
        ids: list[VectorId],
    ) -> int:
        """Delete metadata records synchronously."""
        collection_records = self._records.get(collection, {})
        deleted = 0
        for record_id in ids:
            if record_id in collection_records:
                del collection_records[record_id]
                deleted += 1
        return deleted

    def filter_sync(
        self,
        collection: CollectionName,
        filter_expr: MetadataFilter,
    ) -> list[VectorId]:
        """Return identifiers whose metadata matches the filter expression."""
        collection_records = self._records.get(collection, {})
        return [
            record_id
            for record_id, record in collection_records.items()
            if matches_metadata(record.data, filter_expr)
        ]

    def list_sync(self, collection: CollectionName) -> list[MetadataRecord]:
        """Return all metadata records in a collection."""
        return list(self._records.get(collection, {}).values())

    def get_metadata(
        self,
        collection: CollectionName,
        record_id: VectorId,
    ) -> Metadata | None:
        """Return metadata for a single record, if present."""
        record = self._records.get(collection, {}).get(record_id)
        return record.data if record is not None else None

    async def upsert(self, records: list[MetadataRecord]) -> None:
        self.upsert_sync(records)

    async def get(
        self,
        collection: CollectionName,
        ids: list[VectorId],
    ) -> list[MetadataRecord]:
        return self.get_sync(collection, ids)

    async def delete(
        self,
        collection: CollectionName,
        ids: list[VectorId],
    ) -> int:
        return self.delete_sync(collection, ids)

    async def filter(
        self,
        collection: CollectionName,
        filter_expr: MetadataFilter,
    ) -> list[VectorId]:
        return self.filter_sync(collection, filter_expr)
