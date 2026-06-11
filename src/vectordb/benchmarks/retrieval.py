"""Benchmarks for brute-force vector retrieval."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from vectordb.retrieval.brute_force_search import BruteForceSearcher
from vectordb.retrieval.similarity import SimilarityMetric
from vectordb.storage.models import Document
from vectordb.types import DocumentId


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkConfig:
    """Configuration for a retrieval benchmark run."""

    num_documents: int = 1_000
    dimension: int = 384
    top_k: int = 10
    iterations: int = 20
    metric: SimilarityMetric = SimilarityMetric.COSINE
    seed: int = 42


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkResult:
    """Timing results for brute-force retrieval."""

    num_documents: int
    dimension: int
    top_k: int
    metric: SimilarityMetric
    iterations: int
    total_seconds: float
    mean_seconds: float
    queries_per_second: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_documents": self.num_documents,
            "dimension": self.dimension,
            "top_k": self.top_k,
            "metric": self.metric.value,
            "iterations": self.iterations,
            "total_seconds": self.total_seconds,
            "mean_seconds": self.mean_seconds,
            "queries_per_second": self.queries_per_second,
        }


def _build_documents(config: RetrievalBenchmarkConfig) -> tuple[list[Document], list[float]]:
    rng = np.random.default_rng(config.seed)
    embeddings = rng.standard_normal((config.num_documents, config.dimension))
    query = rng.standard_normal(config.dimension).tolist()

    documents = [
        Document(
            document_id=DocumentId(f"doc-{index}"),
            text=f"document {index}",
            embedding=embeddings[index].astype(float).tolist(),
            metadata={"index": index},
        )
        for index in range(config.num_documents)
    ]
    return documents, query


def run_brute_force_benchmark(
    config: RetrievalBenchmarkConfig | None = None,
) -> RetrievalBenchmarkResult:
    """Benchmark brute-force top-k search latency."""
    resolved = config or RetrievalBenchmarkConfig()
    documents, query = _build_documents(resolved)
    searcher = BruteForceSearcher()

    start = time.perf_counter()
    for _ in range(resolved.iterations):
        searcher.search(
            query_embedding=query,
            documents=documents,
            top_k=resolved.top_k,
            metric=resolved.metric,
        )
    elapsed = time.perf_counter() - start
    mean_seconds = elapsed / resolved.iterations

    return RetrievalBenchmarkResult(
        num_documents=resolved.num_documents,
        dimension=resolved.dimension,
        top_k=resolved.top_k,
        metric=resolved.metric,
        iterations=resolved.iterations,
        total_seconds=elapsed,
        mean_seconds=mean_seconds,
        queries_per_second=resolved.iterations / elapsed if elapsed > 0 else 0.0,
    )
