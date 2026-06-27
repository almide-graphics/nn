"""Recipe: soft_argmax_rows (differentiable argmax / softmax-expected index, src/softargmax.almd)."""
import numpy as np

NAME = "soft_argmax_rows"
MODULE = "softargmax"
CALL = "softargmax.soft_argmax_rows(x)"
TOL = 1e-9
SEED = 20261012


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    z = e.sum(axis=1, keepdims=True)
    idx = np.arange(x.shape[1])
    num = (e * idx).sum(axis=1, keepdims=True)
    return num / z
