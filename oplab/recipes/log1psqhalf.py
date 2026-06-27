"""Recipe: log1p_sq_half (Cauchy NLL ½log(1+x^2), src/log1psqhalf.almd)."""
import numpy as np

NAME = "log1p_sq_half"
MODULE = "log1psqhalf"
CALL = "log1psqhalf.log1p_sq_half(x)"
TOL = 1e-9
SEED = 20260846


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return 0.5 * np.log(1.0 + x * x)
