"""Recipe: recip_1px2_cube (cubed Lorentzian 1/(1+x^2)^3, src/recip1px2cube.almd)."""
import numpy as np

NAME = "recip_1px2_cube"
MODULE = "recip1px2cube"
CALL = "recip1px2cube.recip_1px2_cube(x)"
TOL = 1e-9
SEED = 20260871


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return (1.0 / (1.0 + x * x)) ** 3
