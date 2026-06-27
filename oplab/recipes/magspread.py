"""Recipe: magnitude_spread_rows (per-row magnitude-weighted positional spread, spectral bandwidth, src/magspread.almd)."""
import numpy as np

NAME = "magnitude_spread_rows"
MODULE = "magspread"
CALL = "magspread.magnitude_spread_rows(x)"
TOL = 1e-9
SEED = 20261054


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    a = np.abs(x)
    idx = np.arange(x.shape[1])
    sa = a.sum(axis=1, keepdims=True) + 1e-12
    c = (a * idx).sum(axis=1, keepdims=True) / sa
    return np.sqrt((a * (idx - c) ** 2).sum(axis=1, keepdims=True) / sa)
