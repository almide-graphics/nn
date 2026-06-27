"""Recipe: soft_consensus3_rows (differentiable 3-input consensus/all-agree gate, DLGN, src/softconsensus3.almd)."""
import numpy as np

NAME = "soft_consensus3_rows"
MODULE = "softconsensus3"
CALL = "softconsensus3.soft_consensus3_rows(a, b, c)"
TOL = 1e-9
SEED = 20261022


def make_inputs(rng):
    a = rng.random((5, 4))  # Bernoulli probabilities in [0,1]
    b = rng.random((5, 4))
    c = rng.random((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b), "c": ("matrix", c)}


def reference(a, b, c):
    return a * b * c + (1.0 - a) * (1.0 - b) * (1.0 - c)
