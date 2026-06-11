"""Retrieval pipeline integrating metadata filtering and vector search."""

from dataclasses import dataclass, field

from vectordb.metadata.filters import filter_documents
from vectordb.metadata.metadata_store import InMemoryMetadataStore
from vectordb.retrieval.brute_force_search import BruteForceSearcher, RankedDocument
from vectordb.retrieval.similarity import SimilarityMetric
from vectordb.storage.models import Document
from vectordb.types import CollectionName, DocumentId, MetadataFilter, Vector


@dataclass
class RetrievalPipeline:
    """Apply metadata filters before executing brute-force similarity search."""

    searcher: BruteForceSearcher = field(default_factory=BruteForceSearcher)
    metadata_store: InMemoryMetadataStore | None = None

    def search(
        self,
        query_embedding: Vector,
        documents: list[Document],
        top_k: int = 10,
        metadata_filter: MetadataFilter | None = None,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        collection: CollectionName | None = None,
    ) -> list[RankedDocument]:
        """Filter documents by metadata, then rank the remaining set by similarity."""
        filtered_documents = self._filter_documents(
            documents=documents,
            metadata_filter=metadata_filter,
            collection=collection,
        )
        return self.searcher.search(
            query_embedding=query_embedding,
            documents=filtered_documents,
            top_k=top_k,
            metric=metric,
        )

    def _filter_documents(
        self,
        documents: list[Document],
        metadata_filter: MetadataFilter | None,
        collection: CollectionName | None,
    ) -> list[Document]:
        if metadata_filter is None:
            return documents

        if self.metadata_store is not None and collection is not None:
            allowed_ids = set(self.metadata_store.filter_sync(collection, metadata_filter))
            return [
                document
                for document in documents
                if str(document.document_id) in allowed_ids
            ]

        return filter_documents(documents, metadata_filter)

    def filter_document_ids(
        self,
        document_ids: list[DocumentId],
        metadata_filter: MetadataFilter,
        collection: CollectionName,
    ) -> list[DocumentId]:
        """Return document identifiers allowed by the metadata store filter."""
        if self.metadata_store is None:
            raise ValueError("metadata_store is required for store-backed filtering")

        allowed_ids = set(self.metadata_store.filter_sync(collection, metadata_filter))
        return [
            document_id
            for document_id in document_ids
            if str(document_id) in allowed_ids
        ]
