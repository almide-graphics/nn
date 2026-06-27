"""Recipe: rms_rows (per-row root-mean-square / quadratic mean value, src/rmsrows.almd)."""
import numpy as np

NAME = "rms_rows"
MODULE = "rmsrows"
CALL = "rmsrows.rms_rows(x)"
TOL = 1e-9
SEED = 20260963


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.sqrt((x * x).mean(axis=1, keepdims=True))
