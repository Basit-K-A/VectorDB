"""Tests for vector similarity scoring."""

import numpy as np
import pytest

from vectordb.retrieval.similarity import (
    SimilarityMetric,
    compute_similarity,
    cosine_similarity,
    dot_product_similarity,
    top_k_indices,
)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        scores = cosine_similarity([1.0, 0.0, 0.0], [[1.0, 0.0, 0.0]])
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        scores = cosine_similarity([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0]])
        assert scores[0] == pytest.approx(0.0)
        assert scores[1] == pytest.approx(1.0)

    def test_zero_vector_query(self) -> None:
        scores = cosine_similarity([0.0, 0.0], [[1.0, 2.0]])
        assert scores[0] == pytest.approx(0.0)

    def test_zero_candidate_vector(self) -> None:
        scores = cosine_similarity([1.0, 0.0], [[0.0, 0.0]])
        assert scores[0] == pytest.approx(0.0)


class TestDotProductSimilarity:
    def test_identical_vectors(self) -> None:
        scores = dot_product_similarity([2.0, 3.0], [[2.0, 3.0]])
        assert scores[0] == pytest.approx(13.0)

    def test_orthogonal_vectors(self) -> None:
        scores = dot_product_similarity([1.0, 0.0], [[0.0, 1.0]])
        assert scores[0] == pytest.approx(0.0)


class TestComputeSimilarity:
    def test_dispatcher_uses_cosine(self) -> None:
        scores = compute_similarity([1.0, 0.0], [[1.0, 0.0]], metric=SimilarityMetric.COSINE)
        assert scores[0] == pytest.approx(1.0)

    def test_dispatcher_uses_dot_product(self) -> None:
        scores = compute_similarity([2.0, 1.0], [[2.0, 1.0]], metric=SimilarityMetric.DOT_PRODUCT)
        assert scores[0] == pytest.approx(5.0)

    def test_empty_vectors(self) -> None:
        scores = compute_similarity([1.0, 0.0], [])
        assert scores.size == 0


class TestTopKIndices:
    def test_returns_descending_order(self) -> None:
        scores = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float64)
        indices = top_k_indices(scores, top_k=3)
        assert indices.tolist() == [1, 2, 3]

    def test_top_k_larger_than_collection(self) -> None:
        scores = np.array([0.2, 0.8], dtype=np.float64)
        indices = top_k_indices(scores, top_k=5)
        assert indices.tolist() == [1, 0]

    def test_rejects_invalid_top_k(self) -> None:
        with pytest.raises(ValueError, match="top_k must be greater than zero"):
            top_k_indices(np.array([0.1, 0.2]), top_k=0)
