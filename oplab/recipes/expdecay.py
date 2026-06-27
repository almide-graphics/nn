"""Recipe: exp_decay (Laplace kernel exp(-|x|), src/expdecay.almd)."""
import numpy as np

NAME = "exp_decay"
MODULE = "expdecay"
CALL = "expdecay.exp_decay(x)"
TOL = 1e-9
SEED = 20260782


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-np.abs(x))
