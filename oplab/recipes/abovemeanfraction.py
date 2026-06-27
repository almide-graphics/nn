"""Recipe: above_mean_fraction_rows (per-row fraction of entries above the row mean, nonparametric skew, src/abovemeanfraction.almd)."""
import numpy as np

NAME = "above_mean_fraction_rows"
MODULE = "abovemeanfraction"
CALL = "abovemeanfraction.above_mean_fraction_rows(x)"
TOL = 1e-9
SEED = 20261046


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    mu = x.mean(axis=1, keepdims=True)
    return (x > mu).mean(axis=1, keepdims=True)
