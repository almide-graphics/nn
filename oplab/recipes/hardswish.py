"""Recipe: hardswish (piecewise-linear swish, src/hardswish.almd)."""
import numpy as np

NAME = "hardswish"
MODULE = "hardswish"
CALL = "hardswish.hardswish(x)"
TOL = 1e-9
SEED = 20260717


def make_inputs(rng):
    x = 3.0 * rng.standard_normal((4, 4))   # spread to cover all three regions
    return {"x": ("matrix", x)}


def reference(x):
    return x * np.clip((x + 3.0) / 6.0, 0.0, 1.0)
