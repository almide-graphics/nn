"""Recipe: ternary_scale_rows (per-row TWN scale factor alpha, src/ternaryscale.almd)."""
import numpy as np

NAME = "ternary_scale_rows"
MODULE = "ternaryscale"
CALL = "ternaryscale.ternary_scale_rows(x)"
TOL = 1e-9
SEED = 20260936


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    delta = 0.7 * np.abs(x).mean(axis=1, keepdims=True)
    mask = (np.abs(x) > delta).astype(float)
    above = (np.abs(x) * mask).sum(axis=1, keepdims=True)
    count = mask.sum(axis=1, keepdims=True)
    return np.where(count > 0.0, above / np.where(count > 0.0, count, 1.0), 0.0)
