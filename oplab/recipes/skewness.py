"""Recipe: skewness_rows (population skewness / 3rd standardized moment per row, src/skewness.almd)."""
import numpy as np

NAME = "skewness_rows"
MODULE = "skewness"
CALL = "skewness.skewness_rows(x)"
TOL = 1e-9
SEED = 20260960


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    mu = x.mean(axis=1, keepdims=True)
    dv = x - mu
    m2 = (dv * dv).mean(axis=1, keepdims=True)
    m3 = (dv * dv * dv).mean(axis=1, keepdims=True)
    sigma = np.sqrt(m2)
    return m3 / (sigma * sigma * sigma + 1e-12)
