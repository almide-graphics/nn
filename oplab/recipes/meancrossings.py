"""Recipe: mean_crossings_rows (per-row count of mean crossings, src/meancrossings.almd)."""
import numpy as np

NAME = "mean_crossings_rows"
MODULE = "meancrossings"
CALL = "meancrossings.mean_crossings_rows(x)"
TOL = 1e-9
SEED = 20261003


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    d = x - x.mean(axis=1, keepdims=True)
    s = np.sign(d)
    return (s[:, :-1] * s[:, 1:] < 0).sum(axis=1, keepdims=True).astype(float)
