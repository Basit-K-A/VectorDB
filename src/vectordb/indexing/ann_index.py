"""HNSW-inspired graph index for approximate nearest neighbor search."""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from vectordb.retrieval.similarity import SimilarityMetric, to_float_array
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
    search_depth: int
    num_layers: int
    max_neighbors: int


@dataclass(slots=True)
class GraphNode:
    """A graph node representing one indexed vector."""

    node_id: int
    vector_id: VectorId
    vector: NDArray[np.float64]
    level: int
    neighbors: dict[int, set[int]] = field(default_factory=dict)


class VectorNotFoundError(KeyError):
    """Raised when a vector identifier is not present in the index."""


class HNSWANNIndex:
    """Graph-based hierarchical index inspired by HNSW."""

    def __init__(
        self,
        leafsize: int = 16,
        eps: float = 0.0,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        search_depth: int = 64,
        ef_construction: int = 200,
        seed: int = 42,
    ) -> None:
        if leafsize <= 0:
            raise ValueError("leafsize must be greater than zero")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        if search_depth <= 0:
            raise ValueError("search_depth must be greater than zero")
        if ef_construction <= 0:
            raise ValueError("ef_construction must be greater than zero")

        self._max_neighbors = leafsize
        self._max_neighbors_layer0 = leafsize * 2
        self._eps = eps
        self._metric = metric
        self._search_depth = search_depth
        self._ef_construction = ef_construction
        self._level_multiplier = 1.0 / math.log(max(leafsize, 2))
        self._rng = random.Random(seed)

        self._vectors: dict[VectorId, Vector] = {}
        self._nodes: dict[int, GraphNode] = {}
        self._id_to_node: dict[VectorId, int] = {}
        self._next_node_id = 0
        self._entry_point: int | None = None
        self._max_level = 0
        self._dimension = 0

    @property
    def metric(self) -> SimilarityMetric:
        return self._metric

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def search_depth(self) -> int:
        return self._search_depth

    def __len__(self) -> int:
        return len(self._vectors)

    def insert(self, vector_id: VectorId, vector: Vector) -> None:
        """Insert or replace a vector in the index."""
        if vector_id in self._id_to_node:
            self.remove(vector_id)
        self._insert_new(vector_id, vector)

    def insert_many(self, items: list[tuple[VectorId, Vector]]) -> None:
        """Insert or replace multiple vectors in the index."""
        for vector_id, vector in items:
            self.insert(vector_id, vector)

    def remove(self, vector_id: VectorId) -> None:
        """Remove a vector from the index."""
        node_id = self._id_to_node.pop(vector_id, None)
        if node_id is None:
            raise VectorNotFoundError(f"Vector not found: {vector_id}")

        node = self._nodes.pop(node_id)
        del self._vectors[vector_id]

        for level, neighbors in node.neighbors.items():
            for neighbor_id in neighbors:
                neighbor = self._nodes.get(neighbor_id)
                if neighbor is not None:
                    neighbor.neighbors.get(level, set()).discard(node_id)

        if self._entry_point == node_id:
            self._entry_point = self._choose_new_entry_point()
            self._max_level = max((existing.level for existing in self._nodes.values()), default=0)

    def remove_many(self, vector_ids: list[VectorId]) -> int:
        """Remove multiple vectors. Returns the number removed."""
        removed = 0
        for vector_id in list(vector_ids):
            if vector_id in self._id_to_node:
                self.remove(vector_id)
                removed += 1
        return removed

    def search(self, query_vector: Vector, k: int = 10) -> list[IndexSearchResult]:
        """Return approximate nearest neighbors for a query vector."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not self._nodes or self._entry_point is None:
            return []

        query = to_float_array(self._prepare_vector(query_vector))
        if query.shape[0] != self._dimension:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match index dimension "
                f"{self._dimension}"
            )

        effective_depth = self._effective_search_depth(k)
        entry_point = self._entry_point

        for layer in range(self._max_level, 0, -1):
            nearest = self._search_layer(query, [entry_point], ef=1, layer=layer)
            if nearest:
                entry_point = nearest[0][1]

        candidates = self._search_layer(
            query,
            [entry_point],
            ef=effective_depth,
            layer=0,
        )
        top_k = candidates[: min(k, len(candidates))]

        return [
            IndexSearchResult(
                vector_id=self._nodes[node_id].vector_id,
                distance=distance,
                score=self._distance_to_score(distance),
            )
            for distance, node_id in top_k
        ]

    def stats(self) -> ANNIndexStats:
        """Return index statistics."""
        return ANNIndexStats(
            vector_count=len(self._vectors),
            dimension=self._dimension,
            is_built=self._entry_point is not None,
            leafsize=self._max_neighbors,
            eps=self._eps,
            search_depth=self._search_depth,
            num_layers=self._max_level + 1 if self._nodes else 0,
            max_neighbors=self._max_neighbors,
        )

    def clear(self) -> None:
        """Remove all vectors from the index."""
        self._vectors.clear()
        self._nodes.clear()
        self._id_to_node.clear()
        self._next_node_id = 0
        self._entry_point = None
        self._max_level = 0
        self._dimension = 0

    def _insert_new(self, vector_id: VectorId, vector: Vector) -> None:
        prepared = self._prepare_vector(vector)
        array = to_float_array(prepared)
        if self._dimension == 0:
            self._dimension = int(array.shape[0])
        elif array.shape[0] != self._dimension:
            raise ValueError(
                f"Vector dimension {array.shape[0]} does not match index dimension "
                f"{self._dimension}"
            )

        level = self._assign_level()
        node_id = self._next_node_id
        self._next_node_id += 1

        node = GraphNode(
            node_id=node_id,
            vector_id=vector_id,
            vector=array,
            level=level,
            neighbors={layer_index: set() for layer_index in range(level + 1)},
        )
        self._nodes[node_id] = node
        self._id_to_node[vector_id] = node_id
        self._vectors[vector_id] = prepared

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_level = level
            return

        entry_point = self._entry_point
        for layer in range(self._max_level, level, -1):
            nearest = self._search_layer(array, [entry_point], ef=1, layer=layer)
            if nearest:
                entry_point = nearest[0][1]

        for layer in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(
                array,
                [entry_point],
                ef=self._ef_construction,
                layer=layer,
            )
            max_neighbors = self._layer_capacity(layer)
            neighbors = self._select_neighbors(candidates, max_neighbors)
            for neighbor_id in neighbors:
                self._connect(node_id, neighbor_id, layer)
            if candidates:
                entry_point = candidates[0][1]

        if level > self._max_level:
            self._max_level = level
            self._entry_point = node_id

    def _assign_level(self) -> int:
        uniform = max(self._rng.random(), 1e-9)
        level = int(-math.log(uniform) * self._level_multiplier)
        return min(level, 16)

    def _layer_capacity(self, layer: int) -> int:
        return self._max_neighbors if layer > 0 else self._max_neighbors_layer0

    def _effective_search_depth(self, k: int) -> int:
        scaled = int(self._search_depth * (1.0 - self._eps))
        return max(k, scaled, 1)

    def _distance(self, left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
        diff = left - right
        return float(np.dot(diff, diff))

    def _search_layer(
        self,
        query: NDArray[np.float64],
        entry_points: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        visited: set[int] = set()
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []

        for entry_point in entry_points:
            if entry_point not in self._nodes:
                continue
            distance = self._distance(query, self._nodes[entry_point].vector)
            heapq.heappush(candidates, (distance, entry_point))
            heapq.heappush(results, (-distance, entry_point))
            visited.add(entry_point)

        while candidates:
            distance, current_id = heapq.heappop(candidates)
            farthest_distance = -results[0][0]
            if distance > farthest_distance:
                break

            current_node = self._nodes[current_id]
            for neighbor_id in current_node.neighbors.get(layer, set()):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                neighbor = self._nodes[neighbor_id]
                neighbor_distance = self._distance(query, neighbor.vector)
                if len(results) < ef or neighbor_distance < -results[0][0]:
                    heapq.heappush(candidates, (neighbor_distance, neighbor_id))
                    heapq.heappush(results, (-neighbor_distance, neighbor_id))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted(
            [(-neg_distance, node_id) for neg_distance, node_id in results],
            key=lambda item: item[0],
        )

    def _select_neighbors(
        self,
        candidates: list[tuple[float, int]],
        max_neighbors: int,
    ) -> list[int]:
        selected: list[int] = []
        for _, node_id in candidates:
            if node_id not in self._nodes:
                continue
            selected.append(node_id)
            if len(selected) >= max_neighbors:
                break
        return selected

    def _connect(self, left_id: int, right_id: int, layer: int) -> None:
        left = self._nodes[left_id]
        right = self._nodes[right_id]

        left.neighbors.setdefault(layer, set()).add(right_id)
        right.neighbors.setdefault(layer, set()).add(left_id)

        left_capacity = self._layer_capacity(layer)
        if len(left.neighbors[layer]) > left_capacity:
            self._prune_neighbors(left_id, layer, left_capacity)
        if len(right.neighbors[layer]) > self._layer_capacity(layer):
            self._prune_neighbors(right_id, layer, self._layer_capacity(layer))

    def _prune_neighbors(self, node_id: int, layer: int, capacity: int) -> None:
        node = self._nodes[node_id]
        neighbors = node.neighbors.get(layer, set())
        ranked = sorted(
            (
                (self._distance(node.vector, self._nodes[neighbor].vector), neighbor)
                for neighbor in neighbors
                if neighbor in self._nodes
            ),
            key=lambda item: item[0],
        )
        kept = {neighbor for _, neighbor in ranked[:capacity]}
        for dropped in neighbors - kept:
            node.neighbors[layer].discard(dropped)
            if dropped in self._nodes:
                self._nodes[dropped].neighbors.get(layer, set()).discard(node_id)
        node.neighbors[layer] = kept

    def _choose_new_entry_point(self) -> int | None:
        if not self._nodes:
            return None
        return max(self._nodes.values(), key=lambda node: node.level).node_id

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


# Backward-compatible alias for existing imports and benchmarks.
KDTreeANNIndex = HNSWANNIndex
