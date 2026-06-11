"""Tests for the sentence-transformers embedding encoder."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from vectordb.config import Settings
from vectordb.embeddings.base import EmbeddingRequest
from vectordb.embeddings.encoder import (
    DEFAULT_MODEL,
    EmbeddingCache,
    SentenceTransformerEncoder,
)
from vectordb.types import EmbeddingModelName


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 3
    model.encode.side_effect = lambda texts, convert_to_numpy=True: np.array(
        [[0.1, 0.2, 0.3] for _ in texts],
        dtype=np.float64,
    )
    return model


@pytest.fixture
def encoder(mock_model: MagicMock) -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder(
        model_name=DEFAULT_MODEL,
        cache_size=4,
        enable_cache=True,
        model=mock_model,
    )


class TestEmbeddingCache:
    def test_get_set_and_lru_eviction(self) -> None:
        cache = EmbeddingCache(maxsize=2)
        cache.set("model", "a", [1.0])
        cache.set("model", "b", [2.0])
        cache.set("model", "c", [3.0])

        assert cache.get("model", "a") is None
        assert cache.get("model", "b") == [2.0]
        assert cache.get("model", "c") == [3.0]

    def test_clear(self) -> None:
        cache = EmbeddingCache(maxsize=2)
        cache.set("model", "a", [1.0])
        cache.clear()
        assert len(cache) == 0


class TestSentenceTransformerEncoder:
    def test_default_model_name(self, encoder: SentenceTransformerEncoder) -> None:
        assert encoder.model_name == EmbeddingModelName(DEFAULT_MODEL)
        assert encoder.dimension == 3

    def test_encode_single_text(
        self,
        encoder: SentenceTransformerEncoder,
        mock_model: MagicMock,
    ) -> None:
        vector = encoder.encode("hello world")
        assert vector == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with(["hello world"], convert_to_numpy=True)

    def test_encode_batch(
        self,
        encoder: SentenceTransformerEncoder,
        mock_model: MagicMock,
    ) -> None:
        vectors = encoder.encode_batch(["first", "second"])
        assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        mock_model.encode.assert_called_once_with(["first", "second"], convert_to_numpy=True)

    def test_encode_batch_empty(self, encoder: SentenceTransformerEncoder) -> None:
        assert encoder.encode_batch([]) == []

    def test_cache_avoids_repeat_encoding(
        self,
        encoder: SentenceTransformerEncoder,
        mock_model: MagicMock,
    ) -> None:
        encoder.encode("cached text")
        encoder.encode("cached text")

        mock_model.encode.assert_called_once()

    def test_batch_cache_only_encodes_missing_texts(
        self,
        encoder: SentenceTransformerEncoder,
        mock_model: MagicMock,
    ) -> None:
        encoder.encode("known")
        mock_model.encode.reset_mock()

        vectors = encoder.encode_batch(["known", "new"])
        assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        mock_model.encode.assert_called_once_with(["new"], convert_to_numpy=True)

    def test_cache_can_be_disabled(self, mock_model: MagicMock) -> None:
        encoder = SentenceTransformerEncoder(
            model=mock_model,
            enable_cache=False,
        )
        encoder.encode("repeat")
        encoder.encode("repeat")

        assert mock_model.encode.call_count == 2

    def test_clear_cache(
        self,
        encoder: SentenceTransformerEncoder,
        mock_model: MagicMock,
    ) -> None:
        encoder.encode("repeat")
        encoder.clear_cache()
        encoder.encode("repeat")

        assert mock_model.encode.call_count == 2

    def test_configurable_model_name(self, mock_model: MagicMock) -> None:
        encoder = SentenceTransformerEncoder(
            model_name="custom-model",
            model=mock_model,
        )
        assert encoder.model_name == EmbeddingModelName("custom-model")

    async def test_async_embed(self, encoder: SentenceTransformerEncoder) -> None:
        result = await encoder.embed(EmbeddingRequest(texts=["async one", "async two"]))
        assert result.model == EmbeddingModelName(DEFAULT_MODEL)
        assert result.dimension == 3
        assert len(result.vectors) == 2

    async def test_async_embed_query(self, encoder: SentenceTransformerEncoder) -> None:
        vector = await encoder.embed_query("query text")
        assert vector == [0.1, 0.2, 0.3]

    def test_settings_defaults(self) -> None:
        settings = Settings()
        assert settings.embedding_model == DEFAULT_MODEL
        assert settings.embedding_cache_enabled is True
        assert settings.embedding_cache_size == 1024
