"""Recipe: cumsum_sign_rows (cumulative sum of sign over time, src/cumsumsign.almd)."""
import numpy as np

NAME = "cumsum_sign_rows"
MODULE = "cumsumsign"
CALL = "cumsumsign.cumsum_sign_rows(x)"
TOL = 1e-9
SEED = 20260925


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.cumsum(np.sign(x), axis=0)
