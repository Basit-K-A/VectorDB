"""Exact brute-force vector similarity search."""

from dataclasses import dataclass

from vectordb.metadata.filters import filter_documents
from vectordb.retrieval.similarity import (
    SimilarityMetric,
    compute_similarity,
    stack_vectors,
    to_float_array,
    top_k_indices,
)
from vectordb.storage.models import Document
from vectordb.types import DocumentId, Metadata, MetadataFilter, Score, Vector


@dataclass(frozen=True, slots=True)
class RankedDocument:
    """A document ranked by similarity to a query embedding."""

    document_id: DocumentId
    score: Score
    text: str
    metadata: Metadata


class BruteForceSearcher:
    """Exact top-k search over a document collection using NumPy."""

    def search(
        self,
        query_embedding: Vector,
        documents: list[Document],
        top_k: int = 10,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RankedDocument]:
        """Return the top-k most similar documents for a query embedding."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        filtered_documents = filter_documents(documents, metadata_filter)
        if not filtered_documents:
            return []

        query_array = to_float_array(query_embedding)
        matrix = stack_vectors([document.embedding for document in filtered_documents])
        if matrix.shape[1] != query_array.shape[0]:
            raise ValueError(
                "Query embedding dimension "
                f"{query_array.shape[0]} does not match document dimension {matrix.shape[1]}"
            )

        scores = compute_similarity(query_embedding, matrix, metric=metric)
        ranked_indices = top_k_indices(scores, top_k)

        return [
            RankedDocument(
                document_id=filtered_documents[index].document_id,
                score=float(scores[index]),
                text=filtered_documents[index].text,
                metadata=filtered_documents[index].metadata,
            )
            for index in ranked_indices
        ]

    def search_ids(
        self,
        query_embedding: Vector,
        document_ids: list[DocumentId],
        embeddings: list[Vector],
        top_k: int = 10,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> list[tuple[DocumentId, Score]]:
        """Return ranked document identifiers and scores without full document payloads."""
        if len(document_ids) != len(embeddings):
            raise ValueError("document_ids and embeddings must have the same length")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not document_ids:
            return []

        query_array = to_float_array(query_embedding)
        matrix = stack_vectors(embeddings)
        if matrix.shape[1] != query_array.shape[0]:
            raise ValueError(
                "Query embedding dimension "
                f"{query_array.shape[0]} does not match document dimension {matrix.shape[1]}"
            )

        scores = compute_similarity(query_embedding, matrix, metric=metric)
        ranked_indices = top_k_indices(scores, top_k)
        return [(document_ids[index], float(scores[index])) for index in ranked_indices]
