"""Recipe: tanhshrink (x - tanh(x), src/tanhshrink.almd)."""
import numpy as np

NAME = "tanhshrink"
MODULE = "tanhshrink"
CALL = "tanhshrink.tanhshrink(x)"
TOL = 1e-9
SEED = 20260770


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return x - np.tanh(x)
