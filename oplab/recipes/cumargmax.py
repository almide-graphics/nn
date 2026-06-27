"""Recipe: cumargmax_rows (cumulative argmax over time, running peak position, src/cumargmax.almd)."""
import numpy as np

NAME = "cumargmax_rows"
MODULE = "cumargmax"
CALL = "cumargmax.cumargmax_rows(x)"
TOL = 1e-9
SEED = 20261029


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    out = np.empty((t, d), dtype=float)
    best = x[0].copy()
    bidx = np.zeros(d)
    for i in range(t):
        upd = x[i] > best
        bidx = np.where(upd, float(i), bidx)
        best = np.where(upd, x[i], best)
        out[i] = bidx
    return out
