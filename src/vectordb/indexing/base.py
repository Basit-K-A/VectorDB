"""Vector index interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vectordb.types import CollectionName, VectorId


@dataclass(frozen=True, slots=True)
class IndexBuildRequest:
    """Parameters for building or rebuilding an index."""

    collection: CollectionName
    force_rebuild: bool = False


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Summary statistics for an index."""

    collection: CollectionName
    vector_count: int
    is_built: bool


class Indexer(ABC):
    """Abstract interface for building and maintaining vector indexes."""

    @abstractmethod
    async def build(self, request: IndexBuildRequest) -> IndexStats:
        """Build or rebuild the index for a collection."""

    @abstractmethod
    async def add(self, collection: CollectionName, ids: list[VectorId]) -> None:
        """Add vectors to an existing index."""

    @abstractmethod
    async def remove(self, collection: CollectionName, ids: list[VectorId]) -> None:
        """Remove vectors from an existing index."""

    @abstractmethod
    async def stats(self, collection: CollectionName) -> IndexStats:
        """Return index statistics for a collection."""
