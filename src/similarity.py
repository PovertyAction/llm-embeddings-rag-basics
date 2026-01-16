from __future__ import annotations

import numpy as np


def dot_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Dot product similarity.

    If vectors are normalized, dot product corresponds to cosine similarity.
    """
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    return float(np.dot(a, b))
