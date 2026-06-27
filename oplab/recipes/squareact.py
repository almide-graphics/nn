"""Recipe: square_act (elementwise x^2, src/squareact.almd)."""
import numpy as np

NAME = "square_act"
MODULE = "squareact"
CALL = "squareact.square_act(x)"
TOL = 1e-9
SEED = 20260752


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return x * x
