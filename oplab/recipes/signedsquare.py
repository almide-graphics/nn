"""Recipe: signed_square_rows (sign-preserving square x*|x|, src/signedsquare.almd)."""
import numpy as np

NAME = "signed_square_rows"
MODULE = "signedsquare"
CALL = "signedsquare.signed_square_rows(x)"
TOL = 1e-9
SEED = 20260938


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x * np.abs(x)
