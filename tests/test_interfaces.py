"""Smoke tests ensuring module interfaces are importable."""

import inspect

from vectordb.benchmarks import BenchmarkRunner
from vectordb.embeddings import Embedder
from vectordb.indexing import Indexer
from vectordb.metadata import MetadataStore
from vectordb.query import QueryEngine
from vectordb.retrieval import Retriever
from vectordb.storage import VectorStore


def test_abstract_interfaces_have_required_methods() -> None:
    interfaces = [
        (VectorStore, ["create_collection", "upsert", "get"]),
        (Embedder, ["embed", "embed_query"]),
        (Indexer, ["build", "stats"]),
        (Retriever, ["search"]),
        (MetadataStore, ["upsert", "filter"]),
        (QueryEngine, ["execute"]),
        (BenchmarkRunner, ["run", "list_suites"]),
    ]

    for cls, methods in interfaces:
        for method in methods:
            assert hasattr(cls, method)
            assert inspect.isfunction(getattr(cls, method))
