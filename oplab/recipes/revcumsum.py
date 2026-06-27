"""Recipe: reverse_cumsum_rows (suffix sum over time, src/revcumsum.almd)."""
import numpy as np

NAME = "reverse_cumsum_rows"
MODULE = "revcumsum"
CALL = "revcumsum.reverse_cumsum_rows(x)"
TOL = 1e-9
SEED = 20260899


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.cumsum(x[::-1], axis=0)[::-1]
