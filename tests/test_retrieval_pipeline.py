"""Tests for metadata-aware retrieval pipeline."""

import pytest

from vectordb.metadata.base import MetadataRecord
from vectordb.metadata.metadata_store import InMemoryMetadataStore
from vectordb.retrieval.pipeline import RetrievalPipeline
from vectordb.retrieval.similarity import SimilarityMetric
from vectordb.storage.models import Document
from vectordb.types import DocumentId


def make_document(
    document_id: str,
    embedding: list[float],
    metadata: dict[str, object],
) -> Document:
    return Document(
        document_id=DocumentId(document_id),
        text=f"text-{document_id}",
        embedding=embedding,
        metadata=metadata,
    )


@pytest.fixture
def documents() -> list[Document]:
    return [
        make_document(
            "doc-1",
            [1.0, 0.0],
            {"category": "finance", "author": "john", "year": 2025},
        ),
        make_document(
            "doc-2",
            [0.95, 0.05],
            {"category": "finance", "author": "jane", "year": 2024},
        ),
        make_document(
            "doc-3",
            [0.8, 0.2],
            {"category": "tech", "author": "john", "year": 2023},
        ),
    ]


@pytest.fixture
def metadata_store(documents: list[Document]) -> InMemoryMetadataStore:
    store = InMemoryMetadataStore()
    store.upsert_sync(
        [
            MetadataRecord(
                id=str(document.document_id),
                collection="articles",
                data=document.metadata,
            )
            for document in documents
        ]
    )
    return store


class TestRetrievalPipeline:
    def test_filters_documents_before_search(self, documents: list[Document]) -> None:
        pipeline = RetrievalPipeline()
        results = pipeline.search(
            query_embedding=[1.0, 0.0],
            documents=documents,
            top_k=2,
            metadata_filter={"category": "finance"},
        )

        assert len(results) == 2
        assert all(result.metadata["category"] == "finance" for result in results)

    def test_store_backed_filtering(
        self,
        documents: list[Document],
        metadata_store: InMemoryMetadataStore,
    ) -> None:
        pipeline = RetrievalPipeline(metadata_store=metadata_store)
        results = pipeline.search(
            query_embedding=[1.0, 0.0],
            documents=documents,
            top_k=3,
            metadata_filter={"author": "john"},
            collection="articles",
        )

        assert [result.document_id for result in results] == [
            DocumentId("doc-1"),
            DocumentId("doc-3"),
        ]

    def test_multi_condition_filter_in_pipeline(self, documents: list[Document]) -> None:
        pipeline = RetrievalPipeline()
        results = pipeline.search(
            query_embedding=[1.0, 0.0],
            documents=documents,
            top_k=1,
            metadata_filter={
                "$and": [
                    {"category": "finance"},
                    {"year": {"$gte": 2025}},
                ]
            },
        )

        assert len(results) == 1
        assert results[0].document_id == DocumentId("doc-1")

    def test_no_matches_returns_empty(self, documents: list[Document]) -> None:
        pipeline = RetrievalPipeline()
        results = pipeline.search(
            query_embedding=[1.0, 0.0],
            documents=documents,
            top_k=5,
            metadata_filter={"category": "science"},
        )
        assert results == []

    def test_filter_document_ids(
        self,
        metadata_store: InMemoryMetadataStore,
    ) -> None:
        pipeline = RetrievalPipeline(metadata_store=metadata_store)
        ids = pipeline.filter_document_ids(
            document_ids=[DocumentId("doc-1"), DocumentId("doc-2"), DocumentId("doc-3")],
            metadata_filter={"year": {"$gte": 2024}},
            collection="articles",
        )
        assert ids == [DocumentId("doc-1"), DocumentId("doc-2")]

    def test_brute_force_search_accepts_metadata_filter(self, documents: list[Document]) -> None:
        pipeline = RetrievalPipeline()
        results = pipeline.searcher.search(
            query_embedding=[1.0, 0.0],
            documents=documents,
            top_k=2,
            metric=SimilarityMetric.COSINE,
            metadata_filter={"author": "john"},
        )
        assert len(results) == 2
