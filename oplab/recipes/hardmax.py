"""Recipe: hardmax_rows (one-hot indicator of the row maximum, src/hardmax.almd)."""
import numpy as np

NAME = "hardmax_rows"
MODULE = "hardmax"
CALL = "hardmax.hardmax_rows(x)"
TOL = 1e-9
SEED = 20260992


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    m = x.max(axis=1, keepdims=True)
    return (x >= m).astype(float)
