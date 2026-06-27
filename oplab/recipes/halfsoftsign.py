"""Recipe: half_softsign (½·x/(1+|x|), src/halfsoftsign.almd)."""
import numpy as np

NAME = "half_softsign"
MODULE = "halfsoftsign"
CALL = "halfsoftsign.half_softsign(x)"
TOL = 1e-9
SEED = 20260850


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return 0.5 * x / (1.0 + np.abs(x))
