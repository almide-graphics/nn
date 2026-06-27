"""Recipe: softplus_inverse (log(exp(x)-1), src/softplusinv.almd)."""
import numpy as np

NAME = "softplus_inverse"
MODULE = "softplusinv"
CALL = "softplusinv.softplus_inverse(x)"
TOL = 1e-9
SEED = 20260793


def make_inputs(rng):
    return {"x": ("matrix", rng.uniform(0.5, 3.0, (4, 4)))}  # positive domain


def reference(x):
    return np.log(np.exp(x) - 1.0)
