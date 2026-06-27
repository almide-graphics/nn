"""Recipe: cloglog (complementary log-log 1-exp(-exp(x)), src/cloglog.almd)."""
import numpy as np

NAME = "cloglog"
MODULE = "cloglog"
CALL = "cloglog.cloglog(x)"
TOL = 1e-9
SEED = 20260796


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((4, 4)))}


def reference(x):
    return 1.0 - np.exp(-np.exp(x))
