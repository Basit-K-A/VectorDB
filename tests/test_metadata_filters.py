"""Tests for metadata filter evaluation."""

import pytest

from vectordb.metadata.filters import (
    FilterError,
    filter_documents,
    matches_metadata,
)
from vectordb.storage.models import Document
from vectordb.types import DocumentId


def make_document(document_id: str, metadata: dict[str, object]) -> Document:
    return Document(
        document_id=DocumentId(document_id),
        text=f"text-{document_id}",
        embedding=[1.0, 0.0],
        metadata=metadata,
    )


@pytest.fixture
def documents() -> list[Document]:
    return [
        make_document(
            "finance-john",
            {"category": "finance", "author": "john", "year": 2025},
        ),
        make_document(
            "finance-jane",
            {"category": "finance", "author": "jane", "year": 2024},
        ),
        make_document(
            "tech-john",
            {"category": "tech", "author": "john", "year": 2023},
        ),
    ]


class TestEqualityFilters:
    def test_shorthand_equality(self, documents: list[Document]) -> None:
        filtered = filter_documents(documents, {"category": "finance"})
        assert [doc.document_id for doc in filtered] == [
            DocumentId("finance-john"),
            DocumentId("finance-jane"),
        ]

    def test_explicit_equality(self, documents: list[Document]) -> None:
        filtered = filter_documents(documents, {"author": {"$eq": "john"}})
        assert len(filtered) == 2

    def test_inequality(self, documents: list[Document]) -> None:
        filtered = filter_documents(documents, {"author": {"$ne": "john"}})
        assert [doc.document_id for doc in filtered] == [DocumentId("finance-jane")]

    def test_missing_field_does_not_match(self) -> None:
        metadata = {"category": "finance"}
        assert matches_metadata(metadata, {"author": "john"}) is False


class TestRangeFilters:
    def test_gte_filter(self, documents: list[Document]) -> None:
        filtered = filter_documents(documents, {"year": {"$gte": 2024}})
        assert len(filtered) == 2

    def test_lte_filter(self, documents: list[Document]) -> None:
        filtered = filter_documents(documents, {"year": {"$lte": 2024}})
        assert len(filtered) == 2

    def test_range_filter(self, documents: list[Document]) -> None:
        filtered = filter_documents(documents, {"year": {"$gte": 2024, "$lte": 2025}})
        assert [doc.document_id for doc in filtered] == [
            DocumentId("finance-john"),
            DocumentId("finance-jane"),
        ]

    def test_gt_and_lt(self, documents: list[Document]) -> None:
        filtered = filter_documents(documents, {"year": {"$gt": 2023, "$lt": 2025}})
        assert [doc.document_id for doc in filtered] == [DocumentId("finance-jane")]

    def test_incompatible_types_do_not_match(self) -> None:
        metadata = {"year": "2025"}
        assert matches_metadata(metadata, {"year": {"$gte": 2024}}) is False


class TestMultiConditionFilters:
    def test_implicit_and(self, documents: list[Document]) -> None:
        filtered = filter_documents(
            documents,
            {"category": "finance", "author": "john"},
        )
        assert [doc.document_id for doc in filtered] == [DocumentId("finance-john")]

    def test_explicit_and(self, documents: list[Document]) -> None:
        filtered = filter_documents(
            documents,
            {
                "$and": [
                    {"category": "finance"},
                    {"year": {"$gte": 2025}},
                ]
            },
        )
        assert [doc.document_id for doc in filtered] == [DocumentId("finance-john")]

    def test_or_filter(self, documents: list[Document]) -> None:
        filtered = filter_documents(
            documents,
            {
                "$or": [
                    {"author": "jane"},
                    {"category": "tech"},
                ]
            },
        )
        assert [doc.document_id for doc in filtered] == [
            DocumentId("finance-jane"),
            DocumentId("tech-john"),
        ]

    def test_not_filter(self, documents: list[Document]) -> None:
        filtered = filter_documents(documents, {"$not": {"author": "john"}})
        assert [doc.document_id for doc in filtered] == [DocumentId("finance-jane")]

    def test_empty_filter_matches_all(self, documents: list[Document]) -> None:
        assert filter_documents(documents, {}) == documents

    def test_invalid_and_clause_raises(self) -> None:
        with pytest.raises(FilterError, match="\\$and requires a list"):
            matches_metadata({}, {"$and": {"category": "finance"}})
