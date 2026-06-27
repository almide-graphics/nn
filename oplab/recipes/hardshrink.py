"""Recipe: hardshrink (keep |x|>lambda else 0, src/hardshrink.almd)."""
import numpy as np

NAME = "hardshrink"
MODULE = "hardshrink"
CALL = "hardshrink.hardshrink(x, lambd)"
TOL = 1e-9
SEED = 20260749


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    lambd = float(rng.uniform(0.3, 1.0))
    return {"x": ("matrix", x), "lambd": ("scalar", lambd)}


def reference(x, lambd):
    return np.where(np.abs(x) > lambd, x, 0.0)
