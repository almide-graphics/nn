"""Recipe: autocorr_lag1_rows (per-row lag-1 autocorrelation, src/autocorrlag1.almd)."""
import numpy as np

NAME = "autocorr_lag1_rows"
MODULE = "autocorrlag1"
CALL = "autocorrlag1.autocorr_lag1_rows(x)"
TOL = 1e-9
SEED = 20261001


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    mu = x.mean(axis=1, keepdims=True)
    d = x - mu
    num = (d[:, :-1] * d[:, 1:]).sum(axis=1, keepdims=True)
    den = (d * d).sum(axis=1, keepdims=True)
    return num / (den + 1e-12)
