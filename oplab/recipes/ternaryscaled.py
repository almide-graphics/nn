"""Recipe: ternary_scaled_rows (TWN scaled/dequantized ternary, src/ternaryscaled.almd)."""
import numpy as np

NAME = "ternary_scaled_rows"
MODULE = "ternaryscaled"
CALL = "ternaryscaled.ternary_scaled_rows(x)"
TOL = 1e-9
SEED = 20260924


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    s = np.abs(x).sum(axis=1, keepdims=True)
    d = x.shape[1]
    delta = 0.7 * s / d
    mask = np.abs(x) > delta
    count = mask.sum(axis=1, keepdims=True)
    above = (np.abs(x) * mask).sum(axis=1, keepdims=True)
    alpha = np.where(count > 0, above / np.where(count > 0, count, 1), 0.0)
    t = np.where(x > delta, 1.0, np.where(x < -delta, -1.0, 0.0))
    return alpha * t
