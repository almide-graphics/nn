"""Recipe: decayed_max_rows (leaky/decaying running max, peak-hold envelope, src/decayedmax.almd)."""
import numpy as np

NAME = "decayed_max_rows"
MODULE = "decayedmax"
CALL = "decayedmax.decayed_max_rows(x)"
TOL = 1e-9
SEED = 20261047


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    out = np.empty((t, d), dtype=float)
    m = x[0].copy()
    out[0] = m
    for i in range(1, t):
        m = np.maximum(x[i], 0.9 * m)
        out[i] = m
    return out
