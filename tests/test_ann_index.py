"""Tests for the KD Tree ANN index."""

import numpy as np
import pytest

from vectordb.indexing.ann_index import (
    KDTreeANNIndex,
    VectorNotFoundError,
)
from vectordb.retrieval.similarity import SimilarityMetric


@pytest.fixture
def index() -> KDTreeANNIndex:
    return KDTreeANNIndex(leafsize=4, metric=SimilarityMetric.COSINE)


class TestKDTreeANNIndex:
    def test_insert_and_search(self, index: KDTreeANNIndex) -> None:
        index.insert("a", [1.0, 0.0, 0.0])
        index.insert("b", [0.9, 0.1, 0.0])
        index.insert("c", [0.0, 1.0, 0.0])

        results = index.search([1.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0].vector_id == "a"
        assert results[0].score >= results[1].score

    def test_insert_many(self, index: KDTreeANNIndex) -> None:
        index.insert_many(
            [
                ("a", [1.0, 0.0]),
                ("b", [0.0, 1.0]),
            ]
        )
        assert len(index) == 2
        assert index.stats().is_built is True

    def test_remove_vector(self, index: KDTreeANNIndex) -> None:
        index.insert_many(
            [
                ("a", [1.0, 0.0]),
                ("b", [0.9, 0.1]),
            ]
        )
        index.remove("a")

        results = index.search([1.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0].vector_id == "b"
        assert len(index) == 1

    def test_remove_many(self, index: KDTreeANNIndex) -> None:
        index.insert_many(
            [
                ("a", [1.0, 0.0]),
                ("b", [0.9, 0.1]),
                ("c", [0.0, 1.0]),
            ]
        )
        removed = index.remove_many(["a", "c"])
        assert removed == 2
        assert len(index) == 1

    def test_remove_missing_raises(self, index: KDTreeANNIndex) -> None:
        with pytest.raises(VectorNotFoundError):
            index.remove("missing")

    def test_replace_existing_vector(self, index: KDTreeANNIndex) -> None:
        index.insert("a", [1.0, 0.0])
        index.insert("a", [0.0, 1.0])

        results = index.search([0.0, 1.0], k=1)
        assert results[0].vector_id == "a"
        assert len(index) == 1

    def test_search_empty_index(self, index: KDTreeANNIndex) -> None:
        assert index.search([1.0, 0.0], k=3) == []

    def test_dimension_mismatch_raises(self, index: KDTreeANNIndex) -> None:
        index.insert("a", [1.0, 0.0])
        with pytest.raises(ValueError, match="dimension"):
            index.search([1.0, 0.0, 0.0], k=1)

    def test_invalid_k_raises(self, index: KDTreeANNIndex) -> None:
        index.insert("a", [1.0, 0.0])
        with pytest.raises(ValueError, match="k must be greater than zero"):
            index.search([1.0, 0.0], k=0)

    def test_clear(self, index: KDTreeANNIndex) -> None:
        index.insert("a", [1.0, 0.0])
        index.clear()
        assert len(index) == 0
        assert index.stats().is_built is False

    def test_cosine_normalization_equivalence(self) -> None:
        index = KDTreeANNIndex(metric=SimilarityMetric.COSINE)
        index.insert("a", [3.0, 0.0])
        index.insert("b", [0.0, 4.0])

        results = index.search([1.0, 0.0], k=1)
        assert results[0].vector_id == "a"
        assert results[0].score == pytest.approx(1.0)

    def test_matches_brute_force_top_k_on_small_dataset(self) -> None:
        rng = np.random.default_rng(7)
        vectors = rng.standard_normal((20, 8))
        query = rng.standard_normal(8).tolist()

        index = KDTreeANNIndex(metric=SimilarityMetric.COSINE)
        for idx, vector in enumerate(vectors):
            index.insert(f"doc-{idx}", vector.astype(float).tolist())

        from vectordb.retrieval.brute_force_search import BruteForceSearcher
        from vectordb.storage.models import Document
        from vectordb.types import DocumentId

        documents = [
            Document(
                document_id=DocumentId(f"doc-{idx}"),
                text=str(idx),
                embedding=vector.astype(float).tolist(),
                metadata={},
            )
            for idx, vector in enumerate(vectors)
        ]

        brute = BruteForceSearcher()
        brute_ids = [
            str(result.document_id)
            for result in brute.search(query, documents, top_k=5, metric=SimilarityMetric.COSINE)
        ]
        ann_ids = [result.vector_id for result in index.search(query, k=5)]
        assert ann_ids == brute_ids
