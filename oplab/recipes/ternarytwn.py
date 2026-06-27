"""Recipe: ternary_twn_rows (TWN per-row ternary quantization, src/ternarytwn.almd)."""
import numpy as np

NAME = "ternary_twn_rows"
MODULE = "ternarytwn"
CALL = "ternarytwn.ternary_twn_rows(x)"
TOL = 1e-9
SEED = 20260921


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    delta = 0.7 * np.abs(x).mean(axis=1, keepdims=True)
    return np.where(x > delta, 1.0, np.where(x < -delta, -1.0, 0.0))
