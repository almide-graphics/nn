"""Recipe: chebyshev_dist_rows (row-wise L-inf distance, src/chebyshev.almd)."""
import numpy as np

NAME = "chebyshev_dist_rows"
MODULE = "chebyshev"
CALL = "chebyshev.chebyshev_dist_rows(a, b)"
TOL = 1e-9
SEED = 20260882


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return np.abs(a - b).max(axis=1, keepdims=True)
