"""Recipe: tanh_squared (tanh(x)^2, src/tanhsq.almd)."""
import numpy as np

NAME = "tanh_squared"
MODULE = "tanhsq"
CALL = "tanhsq.tanh_squared(x)"
TOL = 1e-9
SEED = 20260830


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.tanh(x) ** 2
