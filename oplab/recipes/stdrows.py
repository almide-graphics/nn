"""Recipe: std_rows (row-wise population std, src/stdrows.almd)."""
import numpy as np

NAME = "std_rows"
MODULE = "stdrows"
CALL = "stdrows.std_rows(x)"
TOL = 1e-9
SEED = 20260876


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x.std(axis=1, keepdims=True)
