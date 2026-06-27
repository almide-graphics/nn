"""Recipe: demean_sq_rows (squared deviation from the per-channel time mean, src/demeansq.almd)."""
import numpy as np

NAME = "demean_sq_rows"
MODULE = "demeansq"
CALL = "demeansq.demean_sq_rows(x)"
TOL = 1e-9
SEED = 20260916


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return (x - x.mean(axis=0, keepdims=True)) ** 2
