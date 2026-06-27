"""Recipe: l3_norm_rows (per-row L3 / cubic norm value, src/l3norm.almd)."""
import numpy as np

NAME = "l3_norm_rows"
MODULE = "l3norm"
CALL = "l3norm.l3_norm_rows(x)"
TOL = 1e-9
SEED = 20260962


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    a = np.abs(x)
    s = (a * a * a).sum(axis=1, keepdims=True)
    return s ** (1.0 / 3.0)
