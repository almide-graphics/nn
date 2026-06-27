"""Recipe: magnitude_entropy_rows (entropy of L1-normalized magnitude distribution, src/magnitudeentropy.almd)."""
import numpy as np

NAME = "magnitude_entropy_rows"
MODULE = "magnitudeentropy"
CALL = "magnitudeentropy.magnitude_entropy_rows(x)"
TOL = 1e-9
SEED = 20260999


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    a = np.abs(x)
    s = a.sum(axis=1, keepdims=True)
    p = a / (s + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(axis=1, keepdims=True)
