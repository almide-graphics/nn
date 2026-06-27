"""Recipe: threshold (keep x where x>theta else 0, src/threshold.almd)."""
import numpy as np

NAME = "threshold"
MODULE = "threshold"
CALL = "threshold.threshold(x, theta)"
TOL = 1e-9
SEED = 20260742


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    theta = float(rng.uniform(-1.0, 1.0))
    return {"x": ("matrix", x), "theta": ("scalar", theta)}


def reference(x, theta):
    return np.where(x > theta, x, 0.0)
