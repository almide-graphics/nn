"""Recipe: hadamard_ternary_rows (elementwise product of two matrices' TWN ternary codes, src/hadamardternary.almd)."""
import numpy as np

NAME = "hadamard_ternary_rows"
MODULE = "hadamardternary"
CALL = "hadamardternary.hadamard_ternary_rows(a, b)"
TOL = 1e-9
SEED = 20260965


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    da = 0.7 * np.abs(a).mean(axis=1, keepdims=True)
    db = 0.7 * np.abs(b).mean(axis=1, keepdims=True)
    ca = np.where(a > da, 1.0, np.where(a < -da, -1.0, 0.0))
    cb = np.where(b > db, 1.0, np.where(b < -db, -1.0, 0.0))
    return ca * cb
