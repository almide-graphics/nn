"""Recipe: ternary_dot_rows (row-wise TWN ternary inner product, src/ternarydot.almd)."""
import numpy as np

NAME = "ternary_dot_rows"
MODULE = "ternarydot"
CALL = "ternarydot.ternary_dot_rows(a, b)"
TOL = 1e-9
SEED = 20260966


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
    return (ca * cb).sum(axis=1, keepdims=True)
