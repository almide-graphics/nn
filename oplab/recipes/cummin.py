"""Recipe: cummin_rows (cumulative min over time / down rows, src/cummin.almd)."""
import numpy as np

NAME = "cummin_rows"
MODULE = "cummin"
CALL = "cummin.cummin_rows(x)"
TOL = 1e-9
SEED = 20260898


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.minimum.accumulate(x, axis=0)
