"""Recipe: exp_neg_cube_abs (super-Gaussian exp(-|x|^3), src/expnegcubeabs.almd)."""
import numpy as np

NAME = "exp_neg_cube_abs"
MODULE = "expnegcubeabs"
CALL = "expnegcubeabs.exp_neg_cube_abs(x)"
TOL = 1e-9
SEED = 20260833


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-np.abs(x) ** 3)
