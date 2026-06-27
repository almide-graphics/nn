"""Recipe: sqrt_relu (sqrt(max(0,x)), src/sqrtrelu.almd)."""
import numpy as np

NAME = "sqrt_relu"
MODULE = "sqrtrelu"
CALL = "sqrtrelu.sqrt_relu(x)"
TOL = 1e-9
SEED = 20260835


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.sqrt(np.maximum(0.0, x))
