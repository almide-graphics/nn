"""Recipe: signed_sqrt_rows (sign-preserving square root, src/signedsqrt.almd)."""
import numpy as np

NAME = "signed_sqrt_rows"
MODULE = "signedsqrt"
CALL = "signedsqrt.signed_sqrt_rows(x)"
TOL = 1e-9
SEED = 20260929


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.sign(x) * np.sqrt(np.abs(x))
