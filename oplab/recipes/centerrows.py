"""Recipe: center_rows (mean-centering over time / axis=0, src/centerrows.almd)."""
import numpy as np

NAME = "center_rows"
MODULE = "centerrows"
CALL = "centerrows.center_rows(x)"
TOL = 1e-9
SEED = 20260908


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return x - x.mean(axis=0, keepdims=True)
