"""Recipe: sigmoid_squared (sigmoid(x)^2, src/sigmoidsq.almd)."""
import numpy as np

NAME = "sigmoid_squared"
MODULE = "sigmoidsq"
CALL = "sigmoidsq.sigmoid_squared(x)"
TOL = 1e-9
SEED = 20260842


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s ** 2
