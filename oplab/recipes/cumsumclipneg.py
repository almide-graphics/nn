"""Recipe: cumsum_clip_neg_rows (cumulative sum of the negative part over time, src/cumsumclipneg.almd)."""
import numpy as np

NAME = "cumsum_clip_neg_rows"
MODULE = "cumsumclipneg"
CALL = "cumsumclipneg.cumsum_clip_neg_rows(x)"
TOL = 1e-9
SEED = 20260918


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.cumsum(np.minimum(x, 0.0), axis=0)
