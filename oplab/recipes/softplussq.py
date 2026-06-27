"""Recipe: softplus_squared (softplus(x)^2, src/softplussq.almd)."""
import numpy as np

NAME = "softplus_squared"
MODULE = "softplussq"
CALL = "softplussq.softplus_squared(x)"
TOL = 1e-9
SEED = 20260828


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.logaddexp(0.0, x) ** 2
