"""Benchmarks comparing ANN index and brute-force retrieval."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from vectordb.indexing.ann_index import KDTreeANNIndex
from vectordb.retrieval.brute_force_search import BruteForceSearcher
from vectordb.retrieval.similarity import SimilarityMetric
from vectordb.storage.models import Document
from vectordb.types import DocumentId


@dataclass(frozen=True, slots=True)
class IndexBenchmarkConfig:
    """Configuration for ANN vs brute-force benchmarks."""

    num_vectors: int = 1_000
    dimension: int = 32
    top_k: int = 10
    iterations: int = 20
    metric: SimilarityMetric = SimilarityMetric.COSINE
    leafsize: int = 16
    eps: float = 0.0
    seed: int = 42


@dataclass(frozen=True, slots=True)
class IndexBenchmarkResult:
    """Performance metrics for ANN and brute-force search."""

    num_vectors: int
    dimension: int
    top_k: int
    metric: SimilarityMetric
    iterations: int
    brute_force_total_seconds: float
    brute_force_mean_seconds: float
    brute_force_queries_per_second: float
    ann_total_seconds: float
    ann_mean_seconds: float
    ann_queries_per_second: float
    speedup_ratio: float
    recall_at_k: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_vectors": self.num_vectors,
            "dimension": self.dimension,
            "top_k": self.top_k,
            "metric": self.metric.value,
            "iterations": self.iterations,
            "brute_force_total_seconds": self.brute_force_total_seconds,
            "brute_force_mean_seconds": self.brute_force_mean_seconds,
            "brute_force_queries_per_second": self.brute_force_queries_per_second,
            "ann_total_seconds": self.ann_total_seconds,
            "ann_mean_seconds": self.ann_mean_seconds,
            "ann_queries_per_second": self.ann_queries_per_second,
            "speedup_ratio": self.speedup_ratio,
            "recall_at_k": self.recall_at_k,
        }


def _build_dataset(
    config: IndexBenchmarkConfig,
) -> tuple[list[Document], list[tuple[str, list[float]]], list[float]]:
    rng = np.random.default_rng(config.seed)
    embeddings = rng.standard_normal((config.num_vectors, config.dimension))
    query = rng.standard_normal(config.dimension).tolist()

    documents = [
        Document(
            document_id=DocumentId(f"doc-{index}"),
            text=f"document {index}",
            embedding=embeddings[index].astype(float).tolist(),
            metadata={"index": index},
        )
        for index in range(config.num_vectors)
    ]
    items = [
        (f"doc-{index}", documents[index].embedding)
        for index in range(config.num_vectors)
    ]
    return documents, items, query


def _recall_at_k(
    expected_ids: list[str],
    actual_ids: list[str],
) -> float:
    if not expected_ids:
        return 1.0
    expected = set(expected_ids)
    actual = set(actual_ids)
    return len(expected & actual) / len(expected)


def run_index_benchmark(
    config: IndexBenchmarkConfig | None = None,
) -> IndexBenchmarkResult:
    """Benchmark KD Tree ANN search against brute-force search."""
    resolved = config or IndexBenchmarkConfig()
    documents, items, query = _build_dataset(resolved)

    index = KDTreeANNIndex(
        leafsize=resolved.leafsize,
        eps=resolved.eps,
        metric=resolved.metric,
    )
    index.insert_many(items)

    brute_force = BruteForceSearcher()
    brute_results = brute_force.search(
        query_embedding=query,
        documents=documents,
        top_k=resolved.top_k,
        metric=resolved.metric,
    )
    expected_ids = [str(result.document_id) for result in brute_results]
    ann_results = index.search(query, k=resolved.top_k)
    actual_ids = [result.vector_id for result in ann_results]
    recall = _recall_at_k(expected_ids, actual_ids)

    brute_start = time.perf_counter()
    for _ in range(resolved.iterations):
        brute_force.search(
            query_embedding=query,
            documents=documents,
            top_k=resolved.top_k,
            metric=resolved.metric,
        )
    brute_elapsed = time.perf_counter() - brute_start

    ann_start = time.perf_counter()
    for _ in range(resolved.iterations):
        index.search(query, k=resolved.top_k)
    ann_elapsed = time.perf_counter() - ann_start

    brute_mean = brute_elapsed / resolved.iterations
    ann_mean = ann_elapsed / resolved.iterations

    return IndexBenchmarkResult(
        num_vectors=resolved.num_vectors,
        dimension=resolved.dimension,
        top_k=resolved.top_k,
        metric=resolved.metric,
        iterations=resolved.iterations,
        brute_force_total_seconds=brute_elapsed,
        brute_force_mean_seconds=brute_mean,
        brute_force_queries_per_second=resolved.iterations / brute_elapsed
        if brute_elapsed > 0
        else 0.0,
        ann_total_seconds=ann_elapsed,
        ann_mean_seconds=ann_mean,
        ann_queries_per_second=resolved.iterations / ann_elapsed if ann_elapsed > 0 else 0.0,
        speedup_ratio=brute_mean / ann_mean if ann_mean > 0 else 0.0,
        recall_at_k=recall,
    )
