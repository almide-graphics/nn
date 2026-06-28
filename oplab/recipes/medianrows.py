"""Recipe: median_rows (per-row true median via sort-free rank selection, src/medianrows.almd)."""
import numpy as np

NAME = "median_rows"
MODULE = "medianrows"
CALL = "medianrows.median_rows(x)"
TOL = 1e-9
SEED = 20261065


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.median(x, axis=1, keepdims=True)
