"""Recipe: neg_abs (-|x|, src/negabs.almd)."""
import numpy as np

NAME = "neg_abs"
MODULE = "negabs"
CALL = "negabs.neg_abs(x)"
TOL = 1e-9
SEED = 20260823


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return -np.abs(x)
