"""Document models for the storage engine."""

from dataclasses import dataclass

from vectordb.types import DocumentId, Metadata, Vector


@dataclass(frozen=True, slots=True)
class Document:
    """A persisted document with text, embedding, and metadata."""

    document_id: DocumentId
    text: str
    embedding: Vector
    metadata: Metadata
