"""Vector similarity scoring functions."""

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from vectordb.types import Vector

FloatArray = NDArray[np.float64]


class SimilarityMetric(StrEnum):
    """Supported similarity metrics for vector search."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"


def to_float_array(vector: Vector) -> FloatArray:
    """Convert a vector to a contiguous float64 NumPy array."""
    return np.asarray(vector, dtype=np.float64)


def stack_vectors(vectors: list[Vector]) -> FloatArray:
    """Stack vectors into a 2D matrix of shape (n_vectors, dimension)."""
    if not vectors:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack([to_float_array(vector) for vector in vectors])


def cosine_similarity(query: Vector, vectors: list[Vector] | FloatArray) -> FloatArray:
    """Compute cosine similarity between a query and one or more vectors."""
    query_array = to_float_array(query)
    matrix = vectors if isinstance(vectors, np.ndarray) else stack_vectors(vectors)

    if matrix.size == 0:
        return np.empty(0, dtype=np.float64)

    query_norm = np.linalg.norm(query_array)
    if query_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float64)

    vector_norms = np.linalg.norm(matrix, axis=1)
    safe_norms = np.where(vector_norms == 0.0, 1.0, vector_norms)
    scores = matrix @ query_array / (safe_norms * query_norm)
    return np.where(vector_norms == 0.0, 0.0, scores)


def dot_product_similarity(query: Vector, vectors: list[Vector] | FloatArray) -> FloatArray:
    """Compute dot-product similarity between a query and one or more vectors."""
    query_array = to_float_array(query)
    matrix = vectors if isinstance(vectors, np.ndarray) else stack_vectors(vectors)

    if matrix.size == 0:
        return np.empty(0, dtype=np.float64)

    return matrix @ query_array


def compute_similarity(
    query: Vector,
    vectors: list[Vector] | FloatArray,
    metric: SimilarityMetric = SimilarityMetric.COSINE,
) -> FloatArray:
    """Compute similarity scores using the selected metric."""
    if metric == SimilarityMetric.COSINE:
        return cosine_similarity(query, vectors)
    if metric == SimilarityMetric.DOT_PRODUCT:
        return dot_product_similarity(query, vectors)
    raise ValueError(f"Unsupported similarity metric: {metric}")


def top_k_indices(scores: FloatArray, top_k: int) -> NDArray[np.intp]:
    """Return indices of the top-k highest scores in descending order."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if scores.size == 0:
        return np.empty(0, dtype=np.intp)

    k = min(top_k, scores.size)
    if k == scores.size:
        return np.argsort(scores)[::-1]

    candidate_indices = np.argpartition(scores, -k)[-k:]
    return candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
