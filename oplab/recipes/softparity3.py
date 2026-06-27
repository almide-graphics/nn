"""Recipe: soft_parity3_rows (differentiable 3-input parity / XOR3 gate, DLGN, src/softparity3.almd)."""
import numpy as np

NAME = "soft_parity3_rows"
MODULE = "softparity3"
CALL = "softparity3.soft_parity3_rows(a, b, c)"
TOL = 1e-9
SEED = 20261021


def make_inputs(rng):
    a = rng.random((5, 4))  # Bernoulli probabilities in [0,1]
    b = rng.random((5, 4))
    c = rng.random((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b), "c": ("matrix", c)}


def reference(a, b, c):
    return a + b + c - 2.0 * (a * b + b * c + c * a) + 4.0 * a * b * c
