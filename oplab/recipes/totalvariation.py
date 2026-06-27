"""Recipe: total_variation_rows (per-row total variation / L1 path length, src/totalvariation.almd)."""
import numpy as np

NAME = "total_variation_rows"
MODULE = "totalvariation"
CALL = "totalvariation.total_variation_rows(x)"
TOL = 1e-9
SEED = 20261057


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.abs(np.diff(x, axis=1)).sum(axis=1, keepdims=True)
