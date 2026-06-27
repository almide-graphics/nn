"""Recipe: recip_1px4 (flat-top bump 1/(1+x^4), src/recip1px4.almd)."""
import numpy as np

NAME = "recip_1px4"
MODULE = "recip1px4"
CALL = "recip1px4.recip_1px4(x)"
TOL = 1e-9
SEED = 20260824


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return 1.0 / (1.0 + x ** 4)
