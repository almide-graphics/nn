"""Recipe: diff_rows (first difference over time, src/diffrows.almd)."""
import numpy as np

NAME = "diff_rows"
MODULE = "diffrows"
CALL = "diffrows.diff_rows(x)"
TOL = 1e-9
SEED = 20260904


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.vstack([np.zeros((1, x.shape[1])), np.diff(x, axis=0)])
