"""Recipe: second_diff_rows (discrete second difference over time, src/seconddiff.almd)."""
import numpy as np

NAME = "second_diff_rows"
MODULE = "seconddiff"
CALL = "seconddiff.second_diff_rows(x)"
TOL = 1e-9
SEED = 20260976


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    out = np.zeros_like(x, dtype=float)
    out[2:] = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return out
