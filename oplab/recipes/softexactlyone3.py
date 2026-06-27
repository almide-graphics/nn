"""Recipe: soft_exactly_one3_rows (differentiable exactly-one-of-three / one-hot detector, DLGN, src/softexactlyone3.almd)."""
import numpy as np

NAME = "soft_exactly_one3_rows"
MODULE = "softexactlyone3"
CALL = "softexactlyone3.soft_exactly_one3_rows(a, b, c)"
TOL = 1e-9
SEED = 20261023


def make_inputs(rng):
    a = rng.random((5, 4))  # Bernoulli probabilities in [0,1]
    b = rng.random((5, 4))
    c = rng.random((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b), "c": ("matrix", c)}


def reference(a, b, c):
    return a * (1.0 - b) * (1.0 - c) + (1.0 - a) * b * (1.0 - c) + (1.0 - a) * (1.0 - b) * c
