"""Query orchestration interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vectordb.retrieval.base import RetrievalResult
from vectordb.types import CollectionName, MetadataFilter


@dataclass(frozen=True, slots=True)
class QueryRequest:
    """A natural-language or structured query against a collection."""

    collection: CollectionName
    query: str
    top_k: int = 10
    metadata_filter: MetadataFilter | None = None
    include_metadata: bool = True


@dataclass(frozen=True, slots=True)
class QueryResponse:
    """Results from a high-level query."""

    query: str
    results: list[RetrievalResult]
    total: int


class QueryEngine(ABC):
    """Abstract interface for end-to-end query execution."""

    @abstractmethod
    async def execute(self, request: QueryRequest) -> QueryResponse:
        """Embed, search, and return ranked results for a query."""
