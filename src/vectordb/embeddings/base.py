"""Embedding provider interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vectordb.types import EmbeddingModelName, Vector


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """Input payload for embedding generation."""

    texts: list[str]
    model: EmbeddingModelName | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Output from an embedding provider."""

    vectors: list[Vector]
    model: EmbeddingModelName
    dimension: int


class Embedder(ABC):
    """Abstract interface for converting text into dense vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the output dimensionality of produced embeddings."""

    @property
    @abstractmethod
    def model_name(self) -> EmbeddingModelName:
        """Return the default model identifier."""

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate embeddings for one or more text inputs."""

    @abstractmethod
    async def embed_query(self, text: str) -> Vector:
        """Generate a single query embedding optimized for retrieval."""
