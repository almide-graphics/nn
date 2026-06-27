"""Recipe: cosm1 (cos(x)-1, src/cosm1.almd)."""
import numpy as np

NAME = "cosm1"
MODULE = "cosm1"
CALL = "cosm1.cosm1(x)"
TOL = 1e-9
SEED = 20260819


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.cos(x) - 1.0
