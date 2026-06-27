"""Recipe: ternary_mask_rows (nonzero support mask of TWN ternary quantization, src/ternarymask.almd)."""
import numpy as np

NAME = "ternary_mask_rows"
MODULE = "ternarymask"
CALL = "ternarymask.ternary_mask_rows(x)"
TOL = 1e-9
SEED = 20260930


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    delta = 0.7 * np.abs(x).mean(axis=1, keepdims=True)
    return (np.abs(x) > delta).astype(float)
