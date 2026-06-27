"""Recipe: hardsigmoid (piecewise-linear sigmoid, src/hardsigmoid.almd)."""
import numpy as np

NAME = "hardsigmoid"
MODULE = "hardsigmoid"
CALL = "hardsigmoid.hardsigmoid(x)"
TOL = 1e-9
SEED = 20260726


def make_inputs(rng):
    x = 3.0 * rng.standard_normal((4, 4))
    return {"x": ("matrix", x)}


def reference(x):
    return np.clip((x + 3.0) / 6.0, 0.0, 1.0)
