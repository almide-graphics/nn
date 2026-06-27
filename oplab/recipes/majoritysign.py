"""Recipe: majority_sign_rows (sign of the row sum = majority-vote direction, src/majoritysign.almd)."""
import numpy as np

NAME = "majority_sign_rows"
MODULE = "majoritysign"
CALL = "majoritysign.majority_sign_rows(x)"
TOL = 1e-9
SEED = 20260949


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.sign(x.sum(axis=1, keepdims=True))
