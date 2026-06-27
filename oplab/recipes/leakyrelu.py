"""Recipe: leaky_relu (leaky ReLU, src/leakyrelu.almd)."""
import numpy as np

NAME = "leaky_relu"
MODULE = "leakyrelu"
CALL = "leakyrelu.leaky_relu(x, alpha)"
TOL = 1e-9
SEED = 20260737


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    alpha = float(rng.uniform(0.01, 0.3))
    return {"x": ("matrix", x), "alpha": ("scalar", alpha)}


def reference(x, alpha):
    return np.where(x > 0.0, x, alpha * x)
