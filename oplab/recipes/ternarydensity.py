"""Recipe: ternary_density_rows (kept-weight density of TWN ternary quantization, src/ternarydensity.almd)."""
import numpy as np

NAME = "ternary_density_rows"
MODULE = "ternarydensity"
CALL = "ternarydensity.ternary_density_rows(x)"
TOL = 1e-9
SEED = 20260940


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    delta = 0.7 * np.abs(x).mean(axis=1, keepdims=True)
    return (np.abs(x) > delta).mean(axis=1, keepdims=True)
