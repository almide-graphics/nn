"""Recipe: kurtosis_rows (population excess kurtosis / 4th standardized moment per row, src/kurtosis.almd)."""
import numpy as np

NAME = "kurtosis_rows"
MODULE = "kurtosis"
CALL = "kurtosis.kurtosis_rows(x)"
TOL = 1e-9
SEED = 20260961


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    mu = x.mean(axis=1, keepdims=True)
    dv = x - mu
    dd = dv * dv
    m2 = dd.mean(axis=1, keepdims=True)
    m4 = (dd * dd).mean(axis=1, keepdims=True)
    return m4 / (m2 * m2 + 1e-12) - 3.0
