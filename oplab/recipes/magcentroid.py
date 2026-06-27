"""Recipe: magnitude_centroid_rows (per-row magnitude-weighted index centroid, spectral centroid, src/magcentroid.almd)."""
import numpy as np

NAME = "magnitude_centroid_rows"
MODULE = "magcentroid"
CALL = "magcentroid.magnitude_centroid_rows(x)"
TOL = 1e-9
SEED = 20261053


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    a = np.abs(x)
    idx = np.arange(x.shape[1])
    return (a * idx).sum(axis=1, keepdims=True) / (a.sum(axis=1, keepdims=True) + 1e-12)
