"""Recipe: expm1_act (exp(x)-1, src/expm1act.almd)."""
import numpy as np

NAME = "expm1_act"
MODULE = "expm1act"
CALL = "expm1act.expm1_act(x)"
TOL = 1e-9
SEED = 20260764


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(x) - 1.0   # mirror Almide exp(x)-1 (not np.expm1)
