"""Recipe: relu_squared (squared ReLU, src/relusq.almd)."""
import numpy as np

NAME = "relu_squared"
MODULE = "relusq"
CALL = "relusq.relu_squared(x)"
TOL = 1e-9
SEED = 20260747


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.where(x > 0.0, x * x, 0.0)
