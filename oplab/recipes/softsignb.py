"""Recipe: softsign_beta (steepness-parameterized softsign, src/softsignb.almd)."""
import numpy as np

NAME = "softsign_beta"
MODULE = "softsignb"
CALL = "softsignb.softsign_beta(x, beta)"
TOL = 1e-9
SEED = 20260781


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    beta = float(rng.uniform(0.5, 2.0))
    return {"x": ("matrix", x), "beta": ("scalar", beta)}


def reference(x, beta):
    return x / (1.0 + np.abs(beta * x))
