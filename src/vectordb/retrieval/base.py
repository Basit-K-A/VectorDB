"""Similarity search interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vectordb.types import CollectionName, Metadata, MetadataFilter, Score, Vector, VectorId


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Parameters for a similarity search."""

    collection: CollectionName
    query_vector: Vector
    top_k: int = 10
    metadata_filter: MetadataFilter | None = None
    score_threshold: Score | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A single retrieved vector match."""

    id: VectorId
    score: Score
    metadata: Metadata


class Retriever(ABC):
    """Abstract interface for nearest-neighbor vector search."""

    @abstractmethod
    async def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Find the most similar vectors to a query vector."""
