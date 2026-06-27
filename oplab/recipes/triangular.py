"""Recipe: triangular (Bartlett window max(0,1-|x|), src/triangular.almd)."""
import numpy as np

NAME = "triangular"
MODULE = "triangular"
CALL = "triangular.triangular(x)"
TOL = 1e-9
SEED = 20260795


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.maximum(0.0, 1.0 - np.abs(x))
