"""Recipe: clamp_unit (clamp to [0,1], src/clampunit.almd)."""
import numpy as np

NAME = "clamp_unit"
MODULE = "clampunit"
CALL = "clampunit.clamp_unit(x)"
TOL = 1e-9
SEED = 20260756


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.clip(x, 0.0, 1.0)
