"""Recipe: l1_normalize_rows (row-wise L1 normalization, src/l1norm.almd)."""
import numpy as np

NAME = "l1_normalize_rows"
MODULE = "l1norm"
CALL = "l1norm.l1_normalize_rows(x)"
TOL = 1e-9
SEED = 20260730


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x / (np.abs(x).sum(axis=1, keepdims=True) + 1e-12)
