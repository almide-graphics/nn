"""Recipe: cosh_act (hyperbolic cosine, src/coshact.almd)."""
import numpy as np

NAME = "cosh_act"
MODULE = "coshact"
CALL = "coshact.cosh_act(x)"
TOL = 1e-9
SEED = 20260776


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((4, 4)))}


def reference(x):
    return np.cosh(x)
