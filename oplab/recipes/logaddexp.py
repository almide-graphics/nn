"""Recipe: logaddexp_rows (elementwise log-semiring addition log(e^a+e^b), src/logaddexp.almd)."""
import numpy as np

NAME = "logaddexp_rows"
MODULE = "logaddexp"
CALL = "logaddexp.logaddexp_rows(a, b)"
TOL = 1e-9
SEED = 20261025


def make_inputs(rng):
    a = rng.standard_normal((5, 4))
    b = rng.standard_normal((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b)}


def reference(a, b):
    m = np.maximum(a, b)
    return m + np.log(1.0 + np.exp(-np.abs(a - b)))
