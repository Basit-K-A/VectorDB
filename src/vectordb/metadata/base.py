"""Metadata persistence interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vectordb.types import CollectionName, Metadata, MetadataFilter, VectorId


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    """Metadata associated with a stored vector."""

    id: VectorId
    collection: CollectionName
    data: Metadata


class MetadataStore(ABC):
    """Abstract interface for storing and querying vector metadata."""

    @abstractmethod
    async def upsert(self, records: list[MetadataRecord]) -> None:
        """Insert or update metadata records."""

    @abstractmethod
    async def get(
        self,
        collection: CollectionName,
        ids: list[VectorId],
    ) -> list[MetadataRecord]:
        """Retrieve metadata by vector identifier."""

    @abstractmethod
    async def delete(
        self,
        collection: CollectionName,
        ids: list[VectorId],
    ) -> int:
        """Delete metadata records. Returns the number of deleted records."""

    @abstractmethod
    async def filter(
        self,
        collection: CollectionName,
        filter_expr: MetadataFilter,
    ) -> list[VectorId]:
        """Return vector identifiers matching a metadata filter."""
