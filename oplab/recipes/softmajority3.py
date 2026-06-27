"""Recipe: soft_majority3_rows (differentiable 3-input majority gate, DLGN/threshold logic, src/softmajority3.almd)."""
import numpy as np

NAME = "soft_majority3_rows"
MODULE = "softmajority3"
CALL = "softmajority3.soft_majority3_rows(a, b, c)"
TOL = 1e-9
SEED = 20261020


def make_inputs(rng):
    a = rng.random((5, 4))  # Bernoulli probabilities in [0,1]
    b = rng.random((5, 4))
    c = rng.random((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b), "c": ("matrix", c)}


def reference(a, b, c):
    return a * b + b * c + c * a - 2.0 * a * b * c
