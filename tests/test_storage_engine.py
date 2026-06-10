"""Tests for the document storage engine."""

import struct
from pathlib import Path

import pytest

from vectordb.storage.models import Document
from vectordb.storage.serializer import (
    SerializationError,
    deserialize_document,
    deserialize_documents,
    serialize_document,
    serialize_documents,
)
from vectordb.storage.storage_engine import (
    DocumentExistsError,
    DocumentNotFoundError,
    StorageEngine,
)
from vectordb.types import DocumentId


def make_document(
    document_id: str = "doc-1",
    text: str = "hello world",
    embedding: list[float] | None = None,
    metadata: dict[str, object] | None = None,
) -> Document:
    return Document(
        document_id=DocumentId(document_id),
        text=text,
        embedding=embedding or [0.1, 0.2, 0.3],
        metadata=metadata or {"source": "unit-test"},
    )


class TestSerializer:
    def test_round_trip_single_document(self) -> None:
        document = make_document()
        restored = deserialize_document(serialize_document(document))
        assert restored == document

    def test_round_trip_multiple_documents(self) -> None:
        documents = {
            DocumentId("a"): make_document("a", "first"),
            DocumentId("b"): make_document("b", "second", embedding=[1.0, 2.0]),
        }
        restored = deserialize_documents(serialize_documents(documents))
        assert restored == documents

    def test_rejects_invalid_magic(self) -> None:
        with pytest.raises(SerializationError, match="Invalid magic bytes"):
            deserialize_documents(b"BAD!" + b"\x00" * 10)

    def test_rejects_unsupported_version(self) -> None:
        payload = struct.pack(">4sHII", b"VDB1", 99, 0, 0)
        with pytest.raises(SerializationError, match="Unsupported storage format version"):
            deserialize_documents(payload)

    def test_rejects_non_json_metadata(self) -> None:
        document = make_document(metadata={"bad": object()})
        with pytest.raises(SerializationError, match="Metadata is not JSON-serializable"):
            serialize_document(document)


class TestStorageEngine:
    def test_insert_and_get(self) -> None:
        engine = StorageEngine()
        document = make_document()

        engine.insert(document)
        assert engine.get(document.document_id) == document
        assert engine.count() == 1

    def test_insert_duplicate_raises(self) -> None:
        engine = StorageEngine()
        document = make_document()
        engine.insert(document)

        with pytest.raises(DocumentExistsError):
            engine.insert(document)

    def test_get_missing_raises(self) -> None:
        engine = StorageEngine()
        with pytest.raises(DocumentNotFoundError):
            engine.get(DocumentId("missing"))

    def test_update_document(self) -> None:
        engine = StorageEngine()
        original = make_document()
        updated = make_document(text="updated text", metadata={"source": "updated"})

        engine.insert(original)
        engine.update(updated)

        assert engine.get(original.document_id).text == "updated text"
        assert engine.get(original.document_id).metadata == {"source": "updated"}

    def test_update_missing_raises(self) -> None:
        engine = StorageEngine()
        with pytest.raises(DocumentNotFoundError):
            engine.update(make_document())

    def test_delete_document(self) -> None:
        engine = StorageEngine()
        document = make_document()
        engine.insert(document)

        engine.delete(document.document_id)
        assert engine.count() == 0

        with pytest.raises(DocumentNotFoundError):
            engine.delete(document.document_id)

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        storage_file = tmp_path / "documents.vdb"
        engine = StorageEngine(path=storage_file)
        documents = [
            make_document("doc-1", "alpha", [0.1, 0.2]),
            make_document("doc-2", "beta", [0.3, 0.4], {"tag": "beta"}),
        ]
        for document in documents:
            engine.insert(document)

        engine.save()

        reloaded = StorageEngine.from_disk(storage_file)
        assert reloaded.count() == 2
        assert reloaded.get(DocumentId("doc-1")).text == "alpha"
        assert reloaded.get(DocumentId("doc-2")).metadata == {"tag": "beta"}

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        engine = StorageEngine(path=tmp_path / "missing.vdb")
        with pytest.raises(FileNotFoundError):
            engine.load()

    def test_save_requires_path(self) -> None:
        engine = StorageEngine()
        with pytest.raises(ValueError, match="storage path"):
            engine.save()

    def test_persistence_after_mutation(self, tmp_path: Path) -> None:
        storage_file = tmp_path / "documents.vdb"
        engine = StorageEngine(path=storage_file)
        document = make_document()
        engine.insert(document)
        engine.save()

        engine.update(make_document(text="mutated"))
        engine.delete(document.document_id)
        engine.save()

        reloaded = StorageEngine.from_disk(storage_file)
        assert reloaded.count() == 0
