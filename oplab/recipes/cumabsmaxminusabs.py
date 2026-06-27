"""Recipe: cumabsmax_minus_abs_rows (drawdown from the running magnitude peak, src/cumabsmaxminusabs.almd)."""
import numpy as np

NAME = "cumabsmax_minus_abs_rows"
MODULE = "cumabsmaxminusabs"
CALL = "cumabsmaxminusabs.cumabsmax_minus_abs_rows(x)"
TOL = 1e-9
SEED = 20260932


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.maximum.accumulate(np.abs(x), axis=0) - np.abs(x)
