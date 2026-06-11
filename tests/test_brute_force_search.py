"""Tests for brute-force vector search."""

import pytest

from vectordb.retrieval.brute_force_search import BruteForceSearcher, RankedDocument
from vectordb.retrieval.similarity import SimilarityMetric
from vectordb.storage.models import Document
from vectordb.types import DocumentId


def make_document(
    document_id: str,
    text: str,
    embedding: list[float],
    metadata: dict[str, object] | None = None,
) -> Document:
    return Document(
        document_id=DocumentId(document_id),
        text=text,
        embedding=embedding,
        metadata=metadata or {},
    )


@pytest.fixture
def documents() -> list[Document]:
    return [
        make_document("a", "alpha", [1.0, 0.0]),
        make_document("b", "beta", [0.9, 0.1]),
        make_document("c", "gamma", [0.0, 1.0]),
        make_document("d", "delta", [-1.0, 0.0]),
    ]


@pytest.fixture
def searcher() -> BruteForceSearcher:
    return BruteForceSearcher()


class TestBruteForceSearcher:
    def test_cosine_top_k_ranking(
        self,
        searcher: BruteForceSearcher,
        documents: list[Document],
    ) -> None:
        results = searcher.search(
            query_embedding=[1.0, 0.0],
            documents=documents,
            top_k=2,
            metric=SimilarityMetric.COSINE,
        )

        assert len(results) == 2
        assert results[0].document_id == DocumentId("a")
        assert results[1].document_id == DocumentId("b")
        assert results[0].score >= results[1].score

    def test_dot_product_top_k_ranking(
        self,
        searcher: BruteForceSearcher,
        documents: list[Document],
    ) -> None:
        results = searcher.search(
            query_embedding=[1.0, 0.0],
            documents=documents,
            top_k=3,
            metric=SimilarityMetric.DOT_PRODUCT,
        )

        assert [result.document_id for result in results] == [
            DocumentId("a"),
            DocumentId("b"),
            DocumentId("c"),
        ]

    def test_returns_full_document_fields(
        self,
        searcher: BruteForceSearcher,
        documents: list[Document],
    ) -> None:
        results = searcher.search([1.0, 0.0], documents, top_k=1)
        assert isinstance(results[0], RankedDocument)
        assert results[0].text == "alpha"
        assert results[0].metadata == {}

    def test_empty_documents(self, searcher: BruteForceSearcher) -> None:
        assert searcher.search([1.0, 0.0], [], top_k=5) == []

    def test_dimension_mismatch_raises(
        self,
        searcher: BruteForceSearcher,
        documents: list[Document],
    ) -> None:
        with pytest.raises(ValueError, match="dimension"):
            searcher.search([1.0, 0.0, 0.0], documents, top_k=1)

    def test_invalid_top_k_raises(
        self,
        searcher: BruteForceSearcher,
        documents: list[Document],
    ) -> None:
        with pytest.raises(ValueError, match="top_k must be greater than zero"):
            searcher.search([1.0, 0.0], documents, top_k=0)

    def test_search_ids(
        self,
        searcher: BruteForceSearcher,
        documents: list[Document],
    ) -> None:
        ids = [document.document_id for document in documents]
        embeddings = [document.embedding for document in documents]

        ranked = searcher.search_ids(
            query_embedding=[1.0, 0.0],
            document_ids=ids,
            embeddings=embeddings,
            top_k=2,
        )

        assert ranked[0][0] == DocumentId("a")
        assert ranked[0][1] >= ranked[1][1]

    def test_search_ids_length_mismatch_raises(self, searcher: BruteForceSearcher) -> None:
        with pytest.raises(ValueError, match="same length"):
            searcher.search_ids(
                query_embedding=[1.0, 0.0],
                document_ids=[DocumentId("a")],
                embeddings=[[1.0, 0.0], [0.0, 1.0]],
                top_k=1,
            )
