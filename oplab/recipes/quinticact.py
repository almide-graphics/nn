"""Recipe: quintic_act (elementwise x^5, src/quinticact.almd)."""
import numpy as np

NAME = "quintic_act"
MODULE = "quinticact"
CALL = "quinticact.quintic_act(x)"
TOL = 1e-9
SEED = 20260822


def make_inputs(rng):
    return {"x": ("matrix", 1.2 * rng.standard_normal((4, 4)))}


def reference(x):
    return x ** 5
