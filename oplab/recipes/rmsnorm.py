"""Recipe: rms_normalize_vec (gain-free row-wise RMS norm, src/rmsnorm.almd)."""
import numpy as np

NAME = "rms_normalize_vec"
MODULE = "rmsnorm"
CALL = "rmsnorm.rms_normalize_vec(x)"
TOL = 1e-9
SEED = 20260723


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    ms = (x * x).mean(axis=1, keepdims=True)
    return x / np.sqrt(ms + 1e-6)
