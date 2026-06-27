"""Recipe: softclamp (clamp x to [lo,hi], src/softclamp.almd)."""
import numpy as np

NAME = "softclamp"
MODULE = "softclamp"
CALL = "softclamp.softclamp(x, lo, hi)"
TOL = 1e-9
SEED = 20260750


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    lo = float(-rng.uniform(0.5, 2.0))
    hi = float(rng.uniform(0.5, 2.0))
    return {"x": ("matrix", x), "lo": ("scalar", lo), "hi": ("scalar", hi)}


def reference(x, lo, hi):
    return np.clip(x, lo, hi)
