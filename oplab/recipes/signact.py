"""Recipe: sign_act (elementwise sign, src/signact.almd)."""
import numpy as np

NAME = "sign_act"
MODULE = "signact"
CALL = "signact.sign_act(x)"
TOL = 1e-9
SEED = 20260758


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.sign(x)
