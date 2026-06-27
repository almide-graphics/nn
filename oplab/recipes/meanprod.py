"""Recipe: mean_prod_rows (row-wise mean product, src/meanprod.almd)."""
import numpy as np

NAME = "mean_prod_rows"
MODULE = "meanprod"
CALL = "meanprod.mean_prod_rows(a, b)"
TOL = 1e-9
SEED = 20260891


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return (a * b).mean(axis=1, keepdims=True)
