"""Recipe: multiquadric (Hardy RBF sqrt(1+x^2), src/multiquadric.almd)."""
import numpy as np

NAME = "multiquadric"
MODULE = "multiquadric"
CALL = "multiquadric.multiquadric(x)"
TOL = 1e-9
SEED = 20260787


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.sqrt(1.0 + x * x)
