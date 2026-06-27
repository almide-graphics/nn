"""Recipe: quartic_kernel (Biweight bump (1-x^2)_+^2, src/quartic.almd)."""
import numpy as np

NAME = "quartic_kernel"
MODULE = "quartic"
CALL = "quartic.quartic_kernel(x)"
TOL = 1e-9
SEED = 20260794


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.maximum(0.0, 1.0 - x * x) ** 2
