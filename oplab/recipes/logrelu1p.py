"""Recipe: log_relu1p (log(1+relu(x)), src/logrelu1p.almd)."""
import numpy as np

NAME = "log_relu1p"
MODULE = "logrelu1p"
CALL = "logrelu1p.log_relu1p(x)"
TOL = 1e-9
SEED = 20260840


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.log(1.0 + np.maximum(0.0, x))
