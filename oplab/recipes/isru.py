"""Recipe: isru (Inverse Square Root Unit x/sqrt(1+a*x^2), src/isru.almd)."""
import numpy as np

NAME = "isru"
MODULE = "isru"
CALL = "isru.isru(x, alpha)"
TOL = 1e-9
SEED = 20260790


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    alpha = float(rng.uniform(0.5, 3.0))
    return {"x": ("matrix", x), "alpha": ("scalar", alpha)}


def reference(x, alpha):
    return x / np.sqrt(1.0 + alpha * x * x)
