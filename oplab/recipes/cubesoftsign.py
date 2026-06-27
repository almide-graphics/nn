"""Recipe: cube_softsign (softsign(x)^3, src/cubesoftsign.almd)."""
import numpy as np

NAME = "cube_softsign"
MODULE = "cubesoftsign"
CALL = "cubesoftsign.cube_softsign(x)"
TOL = 1e-9
SEED = 20260847


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return (x / (1.0 + np.abs(x))) ** 3
