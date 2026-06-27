"""Recipe: dilation1d_rows (1-D grayscale morphological dilation, sliding max-3 same-pad, src/dilation1d.almd)."""
import numpy as np

NAME = "dilation1d_rows"
MODULE = "dilation1d"
CALL = "dilation1d.dilation1d_rows(x)"
TOL = 1e-9
SEED = 20261059


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    lo = np.concatenate([x[:, :1], x[:, :-1]], axis=1)   # x[j-1], edge-replicated
    hi = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)   # x[j+1], edge-replicated
    return np.maximum(np.maximum(lo, x), hi)
