"""Recipe: mean_abs_dev_rows (per-row mean absolute deviation from the mean, src/meanabsdev.almd)."""
import numpy as np

NAME = "mean_abs_dev_rows"
MODULE = "meanabsdev"
CALL = "meanabsdev.mean_abs_dev_rows(x)"
TOL = 1e-9
SEED = 20260977


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    mu = x.mean(axis=1, keepdims=True)
    return np.abs(x - mu).mean(axis=1, keepdims=True)
