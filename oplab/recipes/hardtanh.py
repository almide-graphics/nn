"""Recipe: hardtanh (clip(x,-1,1), src/hardtanh.almd)."""
import numpy as np

NAME = "hardtanh"
MODULE = "hardtanh"
CALL = "hardtanh.hardtanh(x)"
TOL = 1e-9
SEED = 20260739


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.clip(x, -1.0, 1.0)
