"""Recipe: erosion1d_rows (1-D grayscale morphological erosion, sliding min-3 same-pad, src/erosion1d.almd)."""
import numpy as np

NAME = "erosion1d_rows"
MODULE = "erosion1d"
CALL = "erosion1d.erosion1d_rows(x)"
TOL = 1e-9
SEED = 20261060


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    lo = np.concatenate([x[:, :1], x[:, :-1]], axis=1)   # x[j-1], edge-replicated
    hi = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)   # x[j+1], edge-replicated
    return np.minimum(np.minimum(lo, x), hi)
