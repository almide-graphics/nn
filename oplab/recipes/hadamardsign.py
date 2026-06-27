"""Recipe: hadamard_sign_rows (elementwise product of two matrices' signs, src/hadamardsign.almd)."""
import numpy as np

NAME = "hadamard_sign_rows"
MODULE = "hadamardsign"
CALL = "hadamardsign.hadamard_sign_rows(a, b)"
TOL = 1e-9
SEED = 20260933


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return np.sign(a) * np.sign(b)
