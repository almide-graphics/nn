"""Recipe: soft_xor_rows (differentiable probabilistic XOR gate, DLGN substrate, src/softxor.almd)."""
import numpy as np

NAME = "soft_xor_rows"
MODULE = "softxor"
CALL = "softxor.soft_xor_rows(a, b)"
TOL = 1e-9
SEED = 20261017


def make_inputs(rng):
    a = rng.random((5, 4))  # Bernoulli probabilities in [0,1]
    b = rng.random((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b)}


def reference(a, b):
    return a + b - 2.0 * a * b
