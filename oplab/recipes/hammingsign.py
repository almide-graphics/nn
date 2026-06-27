"""Recipe: hamming_sign_rows (row-wise binary Hamming distance of two matrices' signs, src/hammingsign.almd)."""
import numpy as np

NAME = "hamming_sign_rows"
MODULE = "hammingsign"
CALL = "hammingsign.hamming_sign_rows(a, b)"
TOL = 1e-9
SEED = 20260939


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return (np.sign(a) != np.sign(b)).sum(axis=1, keepdims=True).astype(float)
