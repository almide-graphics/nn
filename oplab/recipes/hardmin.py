"""Recipe: hardmin_rows (one-hot indicator of the row minimum, src/hardmin.almd)."""
import numpy as np

NAME = "hardmin_rows"
MODULE = "hardmin"
CALL = "hardmin.hardmin_rows(x)"
TOL = 1e-9
SEED = 20260993


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    m = x.min(axis=1, keepdims=True)
    return (x <= m).astype(float)
