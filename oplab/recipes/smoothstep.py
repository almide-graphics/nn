"""Recipe: smoothstep (Hermite S-curve 3c^2-2c^3, src/smoothstep.almd)."""
import numpy as np

NAME = "smoothstep"
MODULE = "smoothstep"
CALL = "smoothstep.smoothstep(x)"
TOL = 1e-9
SEED = 20260807


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((4, 4)))}


def reference(x):
    c = np.clip(x, 0.0, 1.0)
    return c * c * (3.0 - 2.0 * c)
