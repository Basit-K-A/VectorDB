"""Shared type definitions used across vector database modules."""

from typing import Any, NewType

type Vector = list[float]
type VectorId = str
type CollectionName = str
type Score = float
type Metadata = dict[str, Any]
type MetadataFilter = dict[str, Any]

DocumentId = NewType("DocumentId", str)
EmbeddingModelName = NewType("EmbeddingModelName", str)
