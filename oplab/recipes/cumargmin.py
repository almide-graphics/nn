"""Recipe: cumargmin_rows (cumulative argmin over time, running trough position, src/cumargmin.almd)."""
import numpy as np

NAME = "cumargmin_rows"
MODULE = "cumargmin"
CALL = "cumargmin.cumargmin_rows(x)"
TOL = 1e-9
SEED = 20261030


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    out = np.empty((t, d), dtype=float)
    best = x[0].copy()
    bidx = np.zeros(d)
    for i in range(t):
        upd = x[i] < best
        bidx = np.where(upd, float(i), bidx)
        best = np.where(upd, x[i], best)
        out[i] = bidx
    return out
