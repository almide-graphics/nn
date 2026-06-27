"""Recipe: max_prod_rows (row-wise max elementwise product, src/maxprod.almd)."""
import numpy as np

NAME = "max_prod_rows"
MODULE = "maxprod"
CALL = "maxprod.max_prod_rows(a, b)"
TOL = 1e-9
SEED = 20260893


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return (a * b).max(axis=1, keepdims=True)
