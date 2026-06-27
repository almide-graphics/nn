"""Recipe: elementwise_max_rows (elementwise maximum of two matrices, src/emax.almd)."""
import numpy as np

NAME = "elementwise_max_rows"
MODULE = "emax"
CALL = "emax.elementwise_max_rows(a, b)"
TOL = 1e-9
SEED = 20260987


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return np.maximum(a, b)
