"""Recipe: soft_and_not_rows (differentiable inhibition gate A AND NOT B, DLGN substrate, src/softandnot.almd)."""
import numpy as np

NAME = "soft_and_not_rows"
MODULE = "softandnot"
CALL = "softandnot.soft_and_not_rows(a, b)"
TOL = 1e-9
SEED = 20261019


def make_inputs(rng):
    a = rng.random((5, 4))  # Bernoulli probabilities in [0,1]
    b = rng.random((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b)}


def reference(a, b):
    return a * (1.0 - b)
