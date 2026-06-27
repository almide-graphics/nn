"""Recipe: smootherstep (Perlin 6c^5-15c^4+10c^3, src/smootherstep.almd)."""
import numpy as np

NAME = "smootherstep"
MODULE = "smootherstep"
CALL = "smootherstep.smootherstep(x)"
TOL = 1e-9
SEED = 20260808


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((4, 4)))}


def reference(x):
    c = np.clip(x, 0.0, 1.0)
    return c * c * c * (c * (c * 6.0 - 15.0) + 10.0)
