"""Recipe: gaussian_bump_shift (RBF centered at -1, exp(-(x+1)^2), src/gaussbumpshift.almd)."""
import numpy as np

NAME = "gaussian_bump_shift"
MODULE = "gaussbumpshift"
CALL = "gaussbumpshift.gaussian_bump_shift(x)"
TOL = 1e-9
SEED = 20260829


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-(x + 1.0) ** 2)
