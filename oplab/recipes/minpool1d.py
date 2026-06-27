"""Recipe: min_pool1d_k2_rows (1-D min pool, kernel 2 stride 2, downsampling, src/minpool1d.almd)."""
import numpy as np

NAME = "min_pool1d_k2_rows"
MODULE = "minpool1d"
CALL = "minpool1d.min_pool1d_k2_rows(x)"
TOL = 1e-9
SEED = 20261039


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    return x.reshape(t, d // 2, 2).min(axis=2)
