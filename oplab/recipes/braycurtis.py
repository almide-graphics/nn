"""Recipe: bray_curtis_dist_rows (row-wise Bray-Curtis dissimilarity, src/braycurtis.almd)."""
import numpy as np

NAME = "bray_curtis_dist_rows"
MODULE = "braycurtis"
CALL = "braycurtis.bray_curtis_dist_rows(a, b)"
TOL = 1e-9
SEED = 20260981


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    num = np.abs(a - b).sum(axis=1, keepdims=True)
    den = (np.abs(a) + np.abs(b)).sum(axis=1, keepdims=True) + 1e-12
    return num / den
