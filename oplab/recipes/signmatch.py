"""Recipe: sign_match_rows (elementwise sign agreement of two matrices, src/signmatch.almd)."""
import numpy as np

NAME = "sign_match_rows"
MODULE = "signmatch"
CALL = "signmatch.sign_match_rows(a, b)"
TOL = 1e-9
SEED = 20260931


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return (np.sign(a) == np.sign(b)).astype(float)
