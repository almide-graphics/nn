"""Recipe: variance_rows (per-row population variance, src/variance.almd)."""
import numpy as np

NAME = "variance_rows"
MODULE = "variance"
CALL = "variance.variance_rows(x)"
TOL = 1e-9
SEED = 20260729


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x.var(axis=1, keepdims=True)   # ddof=0 (population)
