"""Recipe: soft_or_rows (differentiable probabilistic OR gate, DLGN substrate, src/softor.almd)."""
import numpy as np

NAME = "soft_or_rows"
MODULE = "softor"
CALL = "softor.soft_or_rows(a, b)"
TOL = 1e-9
SEED = 20261018


def make_inputs(rng):
    a = rng.random((5, 4))  # Bernoulli probabilities in [0,1]
    b = rng.random((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b)}


def reference(a, b):
    return a + b - a * b
