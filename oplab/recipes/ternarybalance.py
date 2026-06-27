"""Recipe: ternary_balance_rows (net polarity of TWN ternary code per row, src/ternarybalance.almd)."""
import numpy as np

NAME = "ternary_balance_rows"
MODULE = "ternarybalance"
CALL = "ternarybalance.ternary_balance_rows(x)"
TOL = 1e-9
SEED = 20260957


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    delta = 0.7 * np.abs(x).mean(axis=1, keepdims=True)
    code = np.where(x > delta, 1.0, np.where(x < -delta, -1.0, 0.0))
    return code.mean(axis=1, keepdims=True)
