"""Benchmark interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """A named collection of benchmark scenarios."""

    name: str
    description: str
    config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Results from a benchmark run."""

    suite: str
    metrics: dict[str, float]
    metadata: dict[str, Any]


class BenchmarkRunner(ABC):
    """Abstract interface for running performance benchmarks."""

    @abstractmethod
    async def run(self, suite: BenchmarkSuite) -> BenchmarkReport:
        """Execute a benchmark suite and return aggregated metrics."""

    @abstractmethod
    async def list_suites(self) -> list[BenchmarkSuite]:
        """Return available benchmark suites."""
