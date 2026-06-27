"""Recipe: softplus_beta (temperature softplus, src/softplusbeta.almd)."""
import numpy as np

NAME = "softplus_beta"
MODULE = "softplusbeta"
CALL = "softplusbeta.softplus_beta(x, beta)"
TOL = 1e-9
SEED = 20260768


def make_inputs(rng):
    x = rng.standard_normal((4, 4))
    beta = float(rng.uniform(0.5, 2.0))
    return {"x": ("matrix", x), "beta": ("scalar", beta)}


def reference(x, beta):
    return np.logaddexp(0.0, beta * x) / beta
