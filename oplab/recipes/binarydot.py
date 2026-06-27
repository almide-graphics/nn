"""Recipe: binary_dot_rows (row-wise binary inner product of two matrices' signs, src/binarydot.almd)."""
import numpy as np

NAME = "binary_dot_rows"
MODULE = "binarydot"
CALL = "binarydot.binary_dot_rows(a, b)"
TOL = 1e-9
SEED = 20260934


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return (np.sign(a) * np.sign(b)).sum(axis=1, keepdims=True)
