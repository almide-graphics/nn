"""Recipe: rsqrt (1/sqrt(|x|+1e-6), src/rsqrt.almd)."""
import numpy as np

NAME = "rsqrt"
MODULE = "rsqrt"
CALL = "rsqrt.rsqrt(x)"
TOL = 1e-9
SEED = 20260767


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return 1.0 / np.sqrt(np.abs(x) + 1e-6)
