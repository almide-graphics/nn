"""Recipe: negate (elementwise -x, src/negate.almd)."""
import numpy as np

NAME = "negate"
MODULE = "negate"
CALL = "negate.negate(x)"
TOL = 1e-9
SEED = 20260762


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return -x
