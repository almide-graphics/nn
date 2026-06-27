"""Recipe: ternary_quant_error_rows (residual of TWN ternary quantization, src/ternaryquanterror.almd)."""
import numpy as np

NAME = "ternary_quant_error_rows"
MODULE = "ternaryquanterror"
CALL = "ternaryquanterror.ternary_quant_error_rows(x)"
TOL = 1e-9
SEED = 20260946


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    delta = 0.7 * np.abs(x).mean(axis=1, keepdims=True)
    mask = (np.abs(x) > delta).astype(float)
    above = (np.abs(x) * mask).sum(axis=1, keepdims=True)
    count = mask.sum(axis=1, keepdims=True)
    alpha = np.where(count > 0.0, above / np.where(count > 0.0, count, 1.0), 0.0)
    t = np.where(x > delta, 1.0, np.where(x < -delta, -1.0, 0.0))
    return x - alpha * t
