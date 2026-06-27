"""Recipe: clamp_rows (elementwise clamp to per-element [lo,hi], torch.clamp tensor bounds, src/clamp.almd)."""
import numpy as np

NAME = "clamp_rows"
MODULE = "clamp"
CALL = "clamp.clamp_rows(x, lo, hi)"
TOL = 1e-9
SEED = 20261055


def make_inputs(rng):
    x = rng.standard_normal((5, 4))
    lo = rng.standard_normal((5, 4))
    hi = lo + rng.random((5, 4)) + 0.1  # ensure hi >= lo
    return {"x": ("matrix", x), "lo": ("matrix", lo), "hi": ("matrix", hi)}


def reference(x, lo, hi):
    return np.maximum(lo, np.minimum(hi, x))
