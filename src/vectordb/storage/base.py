"""Vector storage interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vectordb.types import CollectionName, Metadata, Vector, VectorId


@dataclass(frozen=True, slots=True)
class StoredVector:
    """A vector and its associated metadata persisted in storage."""

    id: VectorId
    vector: Vector
    metadata: Metadata


class VectorStore(ABC):
    """Abstract interface for persisting and retrieving vectors."""

    @abstractmethod
    async def create_collection(self, name: CollectionName, dimension: int) -> None:
        """Create a new vector collection."""

    @abstractmethod
    async def delete_collection(self, name: CollectionName) -> None:
        """Remove a vector collection."""

    @abstractmethod
    async def upsert(
        self,
        collection: CollectionName,
        vectors: list[StoredVector],
    ) -> list[VectorId]:
        """Insert or update vectors in a collection."""

    @abstractmethod
    async def delete(
        self,
        collection: CollectionName,
        ids: list[VectorId],
    ) -> int:
        """Delete vectors by identifier. Returns the number of deleted vectors."""

    @abstractmethod
    async def get(
        self,
        collection: CollectionName,
        ids: list[VectorId],
    ) -> list[StoredVector]:
        """Retrieve vectors by identifier."""

    @abstractmethod
    async def count(self, collection: CollectionName) -> int:
        """Return the number of vectors in a collection."""
