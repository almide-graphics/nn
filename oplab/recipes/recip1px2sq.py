"""Recipe: recip_1px2_sq (squared Lorentzian 1/(1+x^2)^2, src/recip1px2sq.almd)."""
import numpy as np

NAME = "recip_1px2_sq"
MODULE = "recip1px2sq"
CALL = "recip1px2sq.recip_1px2_sq(x)"
TOL = 1e-9
SEED = 20260866


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return (1.0 / (1.0 + x * x)) ** 2
