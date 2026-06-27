"""Recipe: swish_beta (parameterized swish x·σ(βx), src/swishb.almd)."""
import numpy as np

NAME = "swish_beta"
MODULE = "swishb"
CALL = "swishb.swish_beta(x, beta)"
TOL = 1e-9
SEED = 20260719


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    beta = float(rng.uniform(0.5, 2.0))     # scalar slope
    return {"x": ("matrix", x), "beta": ("scalar", beta)}


def reference(x, beta):
    return x / (1.0 + np.exp(-beta * x))
