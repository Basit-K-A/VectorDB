"""Binary serialization for stored documents."""

import json
import struct
from typing import Final

from vectordb.storage.models import Document
from vectordb.types import DocumentId, Metadata, Vector

MAGIC: Final[bytes] = b"VDB1"
VERSION: Final[int] = 1

_HEADER_FORMAT: Final[str] = ">4sHI"
_DOCUMENT_COUNT_FORMAT: Final[str] = ">I"
_LENGTH_FORMAT: Final[str] = ">I"


class SerializationError(Exception):
    """Raised when binary data cannot be serialized or deserialized."""


def _encode_bytes(value: bytes) -> bytes:
    return struct.pack(_LENGTH_FORMAT, len(value)) + value


def _decode_bytes(data: bytes, offset: int) -> tuple[bytes, int]:
    (length,) = struct.unpack_from(_LENGTH_FORMAT, data, offset)
    offset += struct.calcsize(_LENGTH_FORMAT)
    end = offset + length
    if end > len(data):
        raise SerializationError("Unexpected end of data while reading byte field")
    return data[offset:end], end


def _encode_vector(embedding: Vector) -> bytes:
    if not embedding:
        return struct.pack(_LENGTH_FORMAT, 0)
    packed = struct.pack(f">{len(embedding)}d", *embedding)
    return struct.pack(_LENGTH_FORMAT, len(embedding)) + packed


def _decode_vector(data: bytes, offset: int) -> tuple[Vector, int]:
    (dimension,) = struct.unpack_from(_LENGTH_FORMAT, data, offset)
    offset += struct.calcsize(_LENGTH_FORMAT)
    if dimension == 0:
        return [], offset

    vector_size = dimension * struct.calcsize("d")
    end = offset + vector_size
    if end > len(data):
        raise SerializationError("Unexpected end of data while reading embedding")

    values = struct.unpack_from(f">{dimension}d", data, offset)
    return list(values), end


def _encode_metadata(metadata: Metadata) -> bytes:
    try:
        encoded = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SerializationError("Metadata is not JSON-serializable") from exc
    return _encode_bytes(encoded)


def _decode_metadata(data: bytes, offset: int) -> tuple[Metadata, int]:
    raw, offset = _decode_bytes(data, offset)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SerializationError("Invalid metadata payload") from exc
    if not isinstance(parsed, dict):
        raise SerializationError("Metadata must deserialize to an object")
    return parsed, offset


def serialize_document(document: Document) -> bytes:
    """Serialize a single document to bytes."""
    return serialize_documents({document.document_id: document})


def deserialize_document(data: bytes) -> Document:
    """Deserialize a single-document payload."""
    documents = deserialize_documents(data)
    if len(documents) != 1:
        raise SerializationError("Expected exactly one document in payload")
    return next(iter(documents.values()))


def serialize_documents(documents: dict[DocumentId, Document]) -> bytes:
    """Serialize a document collection to a binary blob."""
    chunks: list[bytes] = [struct.pack(_HEADER_FORMAT, MAGIC, VERSION, 0)]
    chunks.append(struct.pack(_DOCUMENT_COUNT_FORMAT, len(documents)))

    for document in documents.values():
        document_id = str(document.document_id).encode("utf-8")
        text = document.text.encode("utf-8")
        chunks.extend(
            [
                _encode_bytes(document_id),
                _encode_bytes(text),
                _encode_vector(document.embedding),
                _encode_metadata(document.metadata),
            ]
        )

    return b"".join(chunks)


def deserialize_documents(data: bytes) -> dict[DocumentId, Document]:
    """Deserialize a binary blob into a document collection."""
    if len(data) < struct.calcsize(_HEADER_FORMAT) + struct.calcsize(_DOCUMENT_COUNT_FORMAT):
        raise SerializationError("Data is too short to contain a valid storage payload")

    magic, version, _reserved = struct.unpack_from(_HEADER_FORMAT, data, 0)
    if magic != MAGIC:
        raise SerializationError(f"Invalid magic bytes: {magic!r}")
    if version != VERSION:
        raise SerializationError(f"Unsupported storage format version: {version}")

    offset = struct.calcsize(_HEADER_FORMAT)
    (document_count,) = struct.unpack_from(_DOCUMENT_COUNT_FORMAT, data, offset)
    offset += struct.calcsize(_DOCUMENT_COUNT_FORMAT)

    documents: dict[DocumentId, Document] = {}
    for _ in range(document_count):
        document_id_bytes, offset = _decode_bytes(data, offset)
        text_bytes, offset = _decode_bytes(data, offset)
        embedding, offset = _decode_vector(data, offset)
        metadata, offset = _decode_metadata(data, offset)

        document_id = DocumentId(document_id_bytes.decode("utf-8"))
        text = text_bytes.decode("utf-8")
        documents[document_id] = Document(
            document_id=document_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )

    if offset != len(data):
        raise SerializationError("Trailing bytes detected after document payload")

    return documents
