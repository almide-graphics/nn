"""Recipe: dot_rows (row-wise dot product, src/dotrows.almd)."""
import numpy as np

NAME = "dot_rows"
MODULE = "dotrows"
CALL = "dotrows.dot_rows(a, b)"
TOL = 1e-9
SEED = 20260875


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return (a * b).sum(axis=1, keepdims=True)
