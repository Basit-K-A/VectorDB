"""Sentence-transformers embedding encoder."""

import asyncio
from collections import OrderedDict
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from vectordb.embeddings.base import Embedder, EmbeddingRequest, EmbeddingResult
from vectordb.types import EmbeddingModelName, Vector

DEFAULT_MODEL: str = "all-MiniLM-L6-v2"


class EmbeddingCache:
    """LRU cache for text embeddings keyed by model name and input text."""

    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[tuple[str, str], Vector] = OrderedDict()

    def get(self, model_name: str, text: str) -> Vector | None:
        key = (model_name, text)
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, model_name: str, text: str, vector: Vector) -> None:
        key = (model_name, text)
        self._store[key] = vector
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class SentenceTransformerEncoder(Embedder):
    """Generate dense embeddings using sentence-transformers."""

    def __init__(
        self,
        model_name: str | EmbeddingModelName = DEFAULT_MODEL,
        *,
        cache_size: int = 1024,
        enable_cache: bool = True,
        model: SentenceTransformer | None = None,
    ) -> None:
        resolved = str(model_name)
        self._model_name = EmbeddingModelName(resolved)
        self._model = model or SentenceTransformer(resolved)
        self._enable_cache = enable_cache
        self._cache = EmbeddingCache(maxsize=cache_size)

    @property
    def dimension(self) -> int:
        value = self._model.get_sentence_embedding_dimension()
        if value is None:
            raise RuntimeError("Embedding model did not report a dimension")
        return int(value)

    @property
    def model_name(self) -> EmbeddingModelName:
        return self._model_name

    @property
    def cache_enabled(self) -> bool:
        return self._enable_cache

    def clear_cache(self) -> None:
        """Remove all cached embeddings."""
        self._cache.clear()

    def encode(self, text: str) -> Vector:
        """Generate an embedding for a single text."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[Vector]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        if not self._enable_cache:
            return self._encode_with_model(texts)

        results: list[Vector | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        model_key = str(self._model_name)

        for index, text in enumerate(texts):
            cached = self._cache.get(model_key, text)
            if cached is not None:
                results[index] = cached
            else:
                uncached_indices.append(index)
                uncached_texts.append(text)

        if uncached_texts:
            encoded = self._encode_with_model(uncached_texts)
            for index, text, vector in zip(uncached_indices, uncached_texts, encoded, strict=True):
                self._cache.set(model_key, text, vector)
                results[index] = vector

        return [vector for vector in results if vector is not None]

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate embeddings asynchronously for one or more texts."""
        vectors = await asyncio.to_thread(self.encode_batch, request.texts)
        return EmbeddingResult(
            vectors=vectors,
            model=self._model_name,
            dimension=self.dimension,
        )

    async def embed_query(self, text: str) -> Vector:
        """Generate a single query embedding asynchronously."""
        return await asyncio.to_thread(self.encode, text)

    def _encode_with_model(self, texts: list[str]) -> list[Vector]:
        encoded: Any = self._model.encode(texts, convert_to_numpy=True)
        array = np.asarray(encoded)
        if array.ndim == 1:
            return [array.astype(float).tolist()]
        return [row.astype(float).tolist() for row in array]
