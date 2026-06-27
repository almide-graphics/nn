"""Recipe: soft_exactly_two3_rows (differentiable exactly-two-of-three gate, DLGN symmetric family, src/softexactlytwo3.almd)."""
import numpy as np

NAME = "soft_exactly_two3_rows"
MODULE = "softexactlytwo3"
CALL = "softexactlytwo3.soft_exactly_two3_rows(a, b, c)"
TOL = 1e-9
SEED = 20261024


def make_inputs(rng):
    a = rng.random((5, 4))  # Bernoulli probabilities in [0,1]
    b = rng.random((5, 4))
    c = rng.random((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b), "c": ("matrix", c)}


def reference(a, b, c):
    return a * b * (1.0 - c) + a * (1.0 - b) * c + (1.0 - a) * b * c
