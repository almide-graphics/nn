"""Recipe: softsign_sq (softsign(x)^2, src/softsignsq.almd)."""
import numpy as np

NAME = "softsign_sq"
MODULE = "softsignsq"
CALL = "softsignsq.softsign_sq(x)"
TOL = 1e-9
SEED = 20260868


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return (x / (1.0 + np.abs(x))) ** 2
