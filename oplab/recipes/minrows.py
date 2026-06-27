"""Recipe: min_rows (row-wise minimum, src/minrows.almd)."""
import numpy as np

NAME = "min_rows"
MODULE = "minrows"
CALL = "minrows.min_rows(x)"
TOL = 1e-9
SEED = 20260733


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x.min(axis=1, keepdims=True)
