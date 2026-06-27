"""Recipe: cummax_abs_rows (running L-infinity envelope over time, src/cummaxabs.almd)."""
import numpy as np

NAME = "cummax_abs_rows"
MODULE = "cummaxabs"
CALL = "cummaxabs.cummax_abs_rows(x)"
TOL = 1e-9
SEED = 20260906


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.maximum.accumulate(np.abs(x), axis=0)
