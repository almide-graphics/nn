"""Recipe: max_pool1d_k2_rows (1-D max pool, kernel 2 stride 2, downsampling, src/maxpool1d.almd)."""
import numpy as np

NAME = "max_pool1d_k2_rows"
MODULE = "maxpool1d"
CALL = "maxpool1d.max_pool1d_k2_rows(x)"
TOL = 1e-9
SEED = 20261037


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    return x.reshape(t, d // 2, 2).max(axis=2)
