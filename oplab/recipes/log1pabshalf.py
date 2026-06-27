"""Recipe: log1p_abs_half (½·log(1+|x|), src/log1pabshalf.almd)."""
import numpy as np

NAME = "log1p_abs_half"
MODULE = "log1pabshalf"
CALL = "log1pabshalf.log1p_abs_half(x)"
TOL = 1e-9
SEED = 20260869


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return 0.5 * np.log(1.0 + np.abs(x))
