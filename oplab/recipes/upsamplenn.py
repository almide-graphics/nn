"""Recipe: upsample_nearest_x2_rows (nearest-neighbour upsampling x2, src/upsamplenn.almd)."""
import numpy as np

NAME = "upsample_nearest_x2_rows"
MODULE = "upsamplenn"
CALL = "upsamplenn.upsample_nearest_x2_rows(x)"
TOL = 1e-9
SEED = 20261040


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.repeat(x, 2, axis=1)
