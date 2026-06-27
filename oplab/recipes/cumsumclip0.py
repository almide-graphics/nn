"""Recipe: cumsum_clip0_rows (cumulative sum of the positive part over time, src/cumsumclip0.almd)."""
import numpy as np

NAME = "cumsum_clip0_rows"
MODULE = "cumsumclip0"
CALL = "cumsumclip0.cumsum_clip0_rows(x)"
TOL = 1e-9
SEED = 20260917


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.cumsum(np.maximum(x, 0.0), axis=0)
