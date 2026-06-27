"""Recipe: log_cosh_half (½·log(cosh(x)), src/logcoshhalf.almd)."""
import numpy as np

NAME = "log_cosh_half"
MODULE = "logcoshhalf"
CALL = "logcoshhalf.log_cosh_half(x)"
TOL = 1e-9
SEED = 20260860


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    a = np.abs(x)
    return 0.5 * (a + np.log(1.0 + np.exp(-2.0 * a)) - np.log(2.0))
