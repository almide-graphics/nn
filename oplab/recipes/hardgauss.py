"""Recipe: hard_gaussian (Epanechnikov bump max(0,1-x^2), src/hardgauss.almd)."""
import numpy as np

NAME = "hard_gaussian"
MODULE = "hardgauss"
CALL = "hardgauss.hard_gaussian(x)"
TOL = 1e-9
SEED = 20260780


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.maximum(0.0, 1.0 - x * x)
