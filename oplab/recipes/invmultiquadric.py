"""Recipe: inv_multiquadric (RBF 1/sqrt(1+x^2), src/invmultiquadric.almd)."""
import numpy as np

NAME = "inv_multiquadric"
MODULE = "invmultiquadric"
CALL = "invmultiquadric.inv_multiquadric(x)"
TOL = 1e-9
SEED = 20260788


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return 1.0 / np.sqrt(1.0 + x * x)
