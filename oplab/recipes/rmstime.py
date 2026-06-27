"""Recipe: rms_time_rows (per-channel RMS normalization over time, src/rmstime.almd)."""
import numpy as np

NAME = "rms_time_rows"
MODULE = "rmstime"
CALL = "rmstime.rms_time_rows(x)"
TOL = 1e-9
SEED = 20260913


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x / np.sqrt((x ** 2).mean(axis=0, keepdims=True))
