"""Recipe: avg_pool1d_k2_rows (1-D average pool, kernel 2 stride 2, downsampling, src/avgpool1d.almd)."""
import numpy as np

NAME = "avg_pool1d_k2_rows"
MODULE = "avgpool1d"
CALL = "avgpool1d.avg_pool1d_k2_rows(x)"
TOL = 1e-9
SEED = 20261038


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    return x.reshape(t, d // 2, 2).mean(axis=2)
