"""Recipe: cummin_abs_rows (running lower |x| envelope over time, src/cumminabs.almd)."""
import numpy as np

NAME = "cummin_abs_rows"
MODULE = "cumminabs"
CALL = "cumminabs.cummin_abs_rows(x)"
TOL = 1e-9
SEED = 20260907


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.minimum.accumulate(np.abs(x), axis=0)
