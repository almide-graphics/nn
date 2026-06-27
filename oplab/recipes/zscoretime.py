"""Recipe: zscore_time_rows (per-channel standardization over time, src/zscoretime.almd)."""
import numpy as np

NAME = "zscore_time_rows"
MODULE = "zscoretime"
CALL = "zscoretime.zscore_time_rows(x)"
TOL = 1e-9
SEED = 20260909


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
