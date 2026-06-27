"""Recipe: min_prod_rows (row-wise min elementwise product, src/minprod.almd)."""
import numpy as np

NAME = "min_prod_rows"
MODULE = "minprod"
CALL = "minprod.min_prod_rows(a, b)"
TOL = 1e-9
SEED = 20260894


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return (a * b).min(axis=1, keepdims=True)
