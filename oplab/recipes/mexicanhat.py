"""Recipe: mexican_hat (Ricker wavelet (1-x^2)exp(-x^2/2), src/mexicanhat.almd)."""
import numpy as np

NAME = "mexican_hat"
MODULE = "mexicanhat"
CALL = "mexicanhat.mexican_hat(x)"
TOL = 1e-9
SEED = 20260800


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return (1.0 - x * x) * np.exp(-x * x / 2.0)
