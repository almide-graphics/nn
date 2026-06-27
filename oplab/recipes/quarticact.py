"""Recipe: quartic_act (elementwise x^4, src/quarticact.almd)."""
import numpy as np

NAME = "quartic_act"
MODULE = "quarticact"
CALL = "quarticact.quartic_act(x)"
TOL = 1e-9
SEED = 20260818


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return x ** 4
