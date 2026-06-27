"""Recipe: softsign (x/(1+|x|), src/softsign.almd)."""
import numpy as np

NAME = "softsign"
MODULE = "softsign"
CALL = "softsign.softsign(x)"
TOL = 1e-9
SEED = 20260738


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return x / (1.0 + np.abs(x))
