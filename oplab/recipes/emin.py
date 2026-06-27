"""Recipe: elementwise_min_rows (elementwise minimum of two matrices, src/emin.almd)."""
import numpy as np

NAME = "elementwise_min_rows"
MODULE = "emin"
CALL = "emin.elementwise_min_rows(a, b)"
TOL = 1e-9
SEED = 20260988


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return np.minimum(a, b)
