"""Recipe: squareplus (algebraic softplus (x+sqrt(x^2+4))/2, src/squareplus.almd)."""
import numpy as np

NAME = "squareplus"
MODULE = "squareplus"
CALL = "squareplus.squareplus(x)"
TOL = 1e-9
SEED = 20260748


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return (x + np.sqrt(x * x + 4.0)) / 2.0
