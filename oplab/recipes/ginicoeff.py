"""Recipe: gini_coeff_abs_rows (per-row Gini coefficient of magnitudes, src/ginicoeff.almd)."""
import numpy as np

NAME = "gini_coeff_abs_rows"
MODULE = "ginicoeff"
CALL = "ginicoeff.gini_coeff_abs_rows(x)"
TOL = 1e-9
SEED = 20260982


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    d = x.shape[1]
    v = np.abs(x)
    diff = np.abs(v[:, :, None] - v[:, None, :]).sum(axis=2).sum(axis=1, keepdims=True)
    s = v.sum(axis=1, keepdims=True)
    return diff / (2.0 * d * s + 1e-12)
