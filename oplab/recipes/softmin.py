"""Recipe: softmin_rows (row-wise softmax(-x), src/softmin.almd)."""
import numpy as np

NAME = "softmin_rows"
MODULE = "softmin"
CALL = "softmin.softmin_rows(x)"
TOL = 1e-9
SEED = 20260741


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    z = -x
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)
