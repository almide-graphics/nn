"""Recipe: half_square (L2 loss kernel 0.5*x^2, src/halfsquare.almd)."""
import numpy as np

NAME = "half_square"
MODULE = "halfsquare"
CALL = "halfsquare.half_square(x)"
TOL = 1e-9
SEED = 20260816


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return 0.5 * x * x
