"""Tests for retrieval benchmarks."""

import pytest

from vectordb.benchmarks.retrieval import (
    RetrievalBenchmarkConfig,
    run_brute_force_benchmark,
)
from vectordb.retrieval.similarity import SimilarityMetric


class TestRetrievalBenchmark:
    def test_run_brute_force_benchmark(self) -> None:
        result = run_brute_force_benchmark(
            RetrievalBenchmarkConfig(
                num_documents=100,
                dimension=32,
                top_k=5,
                iterations=3,
                metric=SimilarityMetric.COSINE,
            )
        )

        assert result.num_documents == 100
        assert result.dimension == 32
        assert result.top_k == 5
        assert result.iterations == 3
        assert result.total_seconds > 0.0
        assert result.mean_seconds > 0.0
        assert result.queries_per_second > 0.0

    def test_result_as_dict(self) -> None:
        result = run_brute_force_benchmark(
            RetrievalBenchmarkConfig(
                num_documents=20,
                dimension=8,
                top_k=3,
                iterations=2,
            )
        )
        payload = result.as_dict()
        assert payload["metric"] == SimilarityMetric.COSINE.value
        assert payload["queries_per_second"] > 0.0

    @pytest.mark.parametrize("metric", list(SimilarityMetric))
    def test_benchmark_supports_all_metrics(self, metric: SimilarityMetric) -> None:
        result = run_brute_force_benchmark(
            RetrievalBenchmarkConfig(
                num_documents=50,
                dimension=16,
                top_k=4,
                iterations=2,
                metric=metric,
            )
        )
        assert result.metric == metric
