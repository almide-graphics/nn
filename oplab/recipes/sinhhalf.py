"""Recipe: sinh_half (½·sinh(x), src/sinhhalf.almd)."""
import numpy as np

NAME = "sinh_half"
MODULE = "sinhhalf"
CALL = "sinhhalf.sinh_half(x)"
TOL = 1e-9
SEED = 20260859


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return 0.5 * np.sinh(x)
