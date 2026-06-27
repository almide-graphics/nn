"""Recipe: scale_shift (affine a*x+b, src/scaleshift.almd)."""
import numpy as np

NAME = "scale_shift"
MODULE = "scaleshift"
CALL = "scaleshift.scale_shift(x, a, b)"
TOL = 1e-9
SEED = 20260769


def make_inputs(rng):
    x = rng.standard_normal((4, 4))
    a = float(rng.uniform(-2.0, 2.0))
    b = float(rng.uniform(-2.0, 2.0))
    return {"x": ("matrix", x), "a": ("scalar", a), "b": ("scalar", b)}


def reference(x, a, b):
    return a * x + b
