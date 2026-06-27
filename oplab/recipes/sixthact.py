"""Recipe: sixth_act (elementwise x^6, src/sixthact.almd)."""
import numpy as np

NAME = "sixth_act"
MODULE = "sixthact"
CALL = "sixthact.sixth_act(x)"
TOL = 1e-9
SEED = 20260826


def make_inputs(rng):
    return {"x": ("matrix", 1.2 * rng.standard_normal((4, 4)))}


def reference(x):
    return x ** 6
