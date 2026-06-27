"""Recipe: cosh_minus1 (cosh(x)-1, src/coshm1.almd)."""
import numpy as np

NAME = "cosh_minus1"
MODULE = "coshm1"
CALL = "coshm1.cosh_minus1(x)"
TOL = 1e-9
SEED = 20260856


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.cosh(x) - 1.0
