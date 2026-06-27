"""Recipe: cosine_sq (cos^2(x), src/cosinesq.almd)."""
import numpy as np

NAME = "cosine_sq"
MODULE = "cosinesq"
CALL = "cosinesq.cosine_sq(x)"
TOL = 1e-9
SEED = 20260838


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.cos(x) ** 2
