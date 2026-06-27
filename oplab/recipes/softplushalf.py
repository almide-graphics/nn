"""Recipe: softplus_half (½·softplus(x), src/softplushalf.almd)."""
import numpy as np

NAME = "softplus_half"
MODULE = "softplushalf"
CALL = "softplushalf.softplus_half(x)"
TOL = 1e-9
SEED = 20260852


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return 0.5 * np.logaddexp(0.0, x)
