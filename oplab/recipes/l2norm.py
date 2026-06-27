"""Recipe: l2_normalize (row-wise L2 norm, eps=1e-6, src/l2norm.almd)."""
import numpy as np

NAME = "l2norm"
MODULE = "l2norm"
CALL = "l2norm.l2_normalize(x)"
TOL = 1e-9
SEED = 20260703


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    ss = (x * x).sum(axis=1, keepdims=True)
    return x / np.sqrt(ss + 1e-6)
