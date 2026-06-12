"""Tests for ANN index benchmarks."""

import pytest

from vectordb.benchmarks.indexing import IndexBenchmarkConfig, run_index_benchmark
from vectordb.retrieval.similarity import SimilarityMetric


class TestIndexBenchmark:
    def test_run_index_benchmark(self) -> None:
        result = run_index_benchmark(
            IndexBenchmarkConfig(
                num_vectors=200,
                dimension=16,
                top_k=5,
                iterations=5,
            )
        )

        assert result.num_vectors == 200
        assert result.dimension == 16
        assert result.top_k == 5
        assert result.brute_force_total_seconds > 0.0
        assert result.ann_total_seconds > 0.0
        assert result.brute_force_queries_per_second > 0.0
        assert result.ann_queries_per_second > 0.0
        assert result.speedup_ratio > 0.0
        assert 0.0 <= result.recall_at_k <= 1.0

    def test_result_as_dict(self) -> None:
        result = run_index_benchmark(
            IndexBenchmarkConfig(
                num_vectors=50,
                dimension=8,
                top_k=3,
                iterations=2,
            )
        )
        payload = result.as_dict()
        assert payload["metric"] == SimilarityMetric.COSINE.value
        assert "speedup_ratio" in payload
        assert "recall_at_k" in payload

    def test_exact_recall_on_small_exact_index(self) -> None:
        result = run_index_benchmark(
            IndexBenchmarkConfig(
                num_vectors=30,
                dimension=8,
                top_k=5,
                iterations=2,
                eps=0.0,
                search_depth=128,
            )
        )
        assert result.recall_at_k >= 0.8
