"""Recipe: inv_sqrt1px2 (x/sqrt(1+x^2) algebraic sigmoid, src/invsqrt1px2.almd)."""
import numpy as np

NAME = "inv_sqrt1px2"
MODULE = "invsqrt1px2"
CALL = "invsqrt1px2.inv_sqrt1px2(x)"
TOL = 1e-9
SEED = 20260785


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return x / np.sqrt(1.0 + x * x)
