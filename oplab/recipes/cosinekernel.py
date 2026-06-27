"""Recipe: cosine_kernel (raised-cosine window, src/cosinekernel.almd)."""
import numpy as np

NAME = "cosine_kernel"
MODULE = "cosinekernel"
CALL = "cosinekernel.cosine_kernel(x)"
TOL = 1e-9
SEED = 20260798


def make_inputs(rng):
    # keep |x| away from the boundary 1.0 to avoid ULP branch mismatch
    return {"x": ("matrix", rng.uniform(-0.9, 0.9, (4, 4)))}


def reference(x):
    return np.where(np.abs(x) < 1.0, np.cos(np.pi / 2 * np.abs(x)), 0.0)
