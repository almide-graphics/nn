"""Recipe: signed_geomean_rows (sign-aware geometric mean over each row, src/signedgeomean.almd)."""
import numpy as np

NAME = "signed_geomean_rows"
MODULE = "signedgeomean"
CALL = "signedgeomean.signed_geomean_rows(x)"
TOL = 1e-9
SEED = 20260959


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    sign = np.prod(np.sign(x), axis=1, keepdims=True)
    logmean = np.log(np.abs(x)).mean(axis=1, keepdims=True)
    return sign * np.exp(logmean)
