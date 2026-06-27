"""Recipe: ternary_quant_error_l2_rows (per-row L2 norm of TWN ternary residual, src/ternaryquanterrorl2.almd)."""
import numpy as np

NAME = "ternary_quant_error_l2_rows"
MODULE = "ternaryquanterrorl2"
CALL = "ternaryquanterrorl2.ternary_quant_error_l2_rows(x)"
TOL = 1e-9
SEED = 20260974


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    delta = 0.7 * np.abs(x).mean(axis=1, keepdims=True)
    keep = np.abs(x) > delta
    count = keep.sum(axis=1, keepdims=True)
    above = np.where(keep, np.abs(x), 0.0).sum(axis=1, keepdims=True)
    alpha = above / np.where(count > 0, count, 1.0)
    q = np.where(x > delta, alpha, np.where(x < -delta, -alpha, 0.0))
    e = x - q
    return np.sqrt((e * e).sum(axis=1, keepdims=True))
