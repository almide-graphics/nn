"""Recipe: smooth_l1_rows (row-wise smooth-L1 / Huber loss δ=1, src/smoothl1.almd)."""
import numpy as np

NAME = "smooth_l1_rows"
MODULE = "smoothl1"
CALL = "smoothl1.smooth_l1_rows(a, b)"
TOL = 1e-9
SEED = 20260877


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    e = a - b
    ea = np.abs(e)
    huber = np.where(ea < 1.0, 0.5 * e * e, ea - 0.5)
    return huber.mean(axis=1, keepdims=True)
