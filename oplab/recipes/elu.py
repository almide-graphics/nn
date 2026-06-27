"""Recipe: elu (Exponential Linear Unit, src/elu.almd)."""
import numpy as np

NAME = "elu"
MODULE = "elu"
CALL = "elu.elu(x, alpha)"
TOL = 1e-9
SEED = 20260727


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    alpha = float(rng.uniform(0.5, 2.0))
    return {"x": ("matrix", x), "alpha": ("scalar", alpha)}


def reference(x, alpha):
    return np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))
