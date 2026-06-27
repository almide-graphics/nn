"""Recipe: sqrt1p_minus1 (sqrt(1+|x|)-1, src/sqrt1pm1.almd)."""
import numpy as np

NAME = "sqrt1p_minus1"
MODULE = "sqrt1pm1"
CALL = "sqrt1pm1.sqrt1p_minus1(x)"
TOL = 1e-9
SEED = 20260813


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.sqrt(1.0 + np.abs(x)) - 1.0
