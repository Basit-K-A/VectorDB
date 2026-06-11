"""Metadata storage and filtering layer."""

from vectordb.metadata.base import MetadataRecord, MetadataStore
from vectordb.metadata.filters import (
    FilterError,
    FilterOperator,
    filter_documents,
    matches_metadata,
)
from vectordb.metadata.metadata_store import InMemoryMetadataStore

__all__ = [
    "FilterError",
    "FilterOperator",
    "InMemoryMetadataStore",
    "MetadataRecord",
    "MetadataStore",
    "filter_documents",
    "matches_metadata",
]
