"""KD Tree based approximate nearest neighbor index."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree  # type: ignore[import-untyped]

from vectordb.retrieval.similarity import SimilarityMetric, stack_vectors, to_float_array
from vectordb.types import Score, Vector, VectorId


@dataclass(frozen=True, slots=True)
class IndexSearchResult:
    """A nearest-neighbor search result from the ANN index."""

    vector_id: VectorId
    distance: float
    score: Score


@dataclass(frozen=True, slots=True)
class ANNIndexStats:
    """Summary statistics for the ANN index."""

    vector_count: int
    dimension: int
    is_built: bool
    leafsize: int
    eps: float


class VectorNotFoundError(KeyError):
    """Raised when a vector identifier is not present in the index."""


class KDTreeANNIndex:
    """KD Tree index supporting insert, remove, and nearest-neighbor search."""

    def __init__(
        self,
        leafsize: int = 16,
        eps: float = 0.0,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> None:
        if leafsize <= 0:
            raise ValueError("leafsize must be greater than zero")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")

        self._leafsize = leafsize
        self._eps = eps
        self._metric = metric
        self._vectors: dict[VectorId, Vector] = {}
        self._tree: KDTree | None = None
        self._matrix: NDArray[np.float64] | None = None
        self._ids: list[VectorId] = []

    @property
    def metric(self) -> SimilarityMetric:
        return self._metric

    @property
    def eps(self) -> float:
        return self._eps

    def __len__(self) -> int:
        return len(self._vectors)

    def insert(self, vector_id: VectorId, vector: Vector) -> None:
        """Insert or replace a vector in the index."""
        self._vectors[vector_id] = self._prepare_vector(vector)
        self._rebuild()

    def insert_many(self, items: list[tuple[VectorId, Vector]]) -> None:
        """Insert or replace multiple vectors in the index."""
        for vector_id, vector in items:
            self._vectors[vector_id] = self._prepare_vector(vector)
        self._rebuild()

    def remove(self, vector_id: VectorId) -> None:
        """Remove a vector from the index."""
        if vector_id not in self._vectors:
            raise VectorNotFoundError(f"Vector not found: {vector_id}")
        del self._vectors[vector_id]
        self._rebuild()

    def remove_many(self, vector_ids: list[VectorId]) -> int:
        """Remove multiple vectors. Returns the number removed."""
        removed = 0
        for vector_id in vector_ids:
            if vector_id in self._vectors:
                del self._vectors[vector_id]
                removed += 1
        if removed:
            self._rebuild()
        return removed

    def search(self, query_vector: Vector, k: int = 10) -> list[IndexSearchResult]:
        """Return approximate nearest neighbors for a query vector."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not self._vectors or self._tree is None or self._matrix is None:
            return []

        query = to_float_array(self._prepare_vector(query_vector))
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match index dimension "
                f"{self._matrix.shape[1]}"
            )

        neighbor_count = min(k, len(self._ids))
        distances, indices = self._tree.query(
            query,
            k=neighbor_count,
            eps=self._eps,
            workers=-1,
        )

        if neighbor_count == 1:
            distance_list = [float(distances)]
            index_list = [int(indices)]
        else:
            distance_list = [float(value) for value in distances]
            index_list = [int(value) for value in indices]

        return [
            IndexSearchResult(
                vector_id=self._ids[index],
                distance=distance,
                score=self._distance_to_score(distance),
            )
            for index, distance in zip(index_list, distance_list, strict=True)
        ]

    def stats(self) -> ANNIndexStats:
        """Return index statistics."""
        dimension = 0 if self._matrix is None else int(self._matrix.shape[1])
        return ANNIndexStats(
            vector_count=len(self._vectors),
            dimension=dimension,
            is_built=self._tree is not None,
            leafsize=self._leafsize,
            eps=self._eps,
        )

    def clear(self) -> None:
        """Remove all vectors from the index."""
        self._vectors.clear()
        self._tree = None
        self._matrix = None
        self._ids = []

    def _rebuild(self) -> None:
        if not self._vectors:
            self._tree = None
            self._matrix = None
            self._ids = []
            return

        self._ids = list(self._vectors.keys())
        self._matrix = stack_vectors([self._vectors[vector_id] for vector_id in self._ids])
        self._tree = KDTree(self._matrix, leafsize=self._leafsize)

    def _prepare_vector(self, vector: Vector) -> Vector:
        array = to_float_array(vector)
        if self._metric == SimilarityMetric.COSINE:
            norm = np.linalg.norm(array)
            if norm == 0.0:
                return [float(value) for value in array.tolist()]
            normalized = array / norm
            return [float(value) for value in normalized.tolist()]
        return [float(value) for value in array.tolist()]

    def _distance_to_score(self, distance: float) -> Score:
        if self._metric == SimilarityMetric.COSINE:
            cosine = 1.0 - (distance**2) / 2.0
            return float(max(min(cosine, 1.0), -1.0))
        return float(-distance)
