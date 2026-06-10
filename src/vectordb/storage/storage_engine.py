"""Disk-backed document storage engine."""

from pathlib import Path

from vectordb.storage.models import Document
from vectordb.storage.serializer import (
    SerializationError,
    deserialize_documents,
    serialize_documents,
)
from vectordb.types import DocumentId


class DocumentExistsError(Exception):
    """Raised when inserting a document with an existing identifier."""


class DocumentNotFoundError(Exception):
    """Raised when a document identifier does not exist."""


class StorageEngine:
    """In-memory document store with binary persistence to disk."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._documents: dict[DocumentId, Document] = {}

    @property
    def path(self) -> Path | None:
        """Return the configured persistence path, if any."""
        return self._path

    def insert(self, document: Document) -> None:
        """Insert a new document."""
        if document.document_id in self._documents:
            raise DocumentExistsError(f"Document already exists: {document.document_id}")
        self._documents[document.document_id] = document

    def get(self, document_id: DocumentId) -> Document:
        """Return a document by identifier."""
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise DocumentNotFoundError(f"Document not found: {document_id}") from exc

    def delete(self, document_id: DocumentId) -> None:
        """Delete a document by identifier."""
        if document_id not in self._documents:
            raise DocumentNotFoundError(f"Document not found: {document_id}")
        del self._documents[document_id]

    def update(self, document: Document) -> None:
        """Replace an existing document."""
        if document.document_id not in self._documents:
            raise DocumentNotFoundError(f"Document not found: {document.document_id}")
        self._documents[document.document_id] = document

    def list_ids(self) -> list[DocumentId]:
        """Return all stored document identifiers."""
        return list(self._documents.keys())

    def count(self) -> int:
        """Return the number of stored documents."""
        return len(self._documents)

    def save(self, path: Path | None = None) -> Path:
        """Persist all documents to disk using binary serialization."""
        target = path or self._path
        if target is None:
            raise ValueError("A storage path must be provided")

        target.parent.mkdir(parents=True, exist_ok=True)
        payload = serialize_documents(self._documents)
        target.write_bytes(payload)
        self._path = target
        return target

    def load(self, path: Path | None = None) -> None:
        """Load documents from a binary file on disk."""
        target = path or self._path
        if target is None:
            raise ValueError("A storage path must be provided")
        if not target.exists():
            raise FileNotFoundError(f"Storage file not found: {target}")

        try:
            self._documents = deserialize_documents(target.read_bytes())
        except SerializationError:
            raise
        self._path = target

    @classmethod
    def from_disk(cls, path: Path) -> "StorageEngine":
        """Create a storage engine populated from an on-disk snapshot."""
        engine = cls(path=path)
        engine.load()
        return engine
