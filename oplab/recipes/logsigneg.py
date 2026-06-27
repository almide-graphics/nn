"""Recipe: log_sigmoid_neg (log sigmoid(-x) = -softplus(x), src/logsigneg.almd)."""
import numpy as np

NAME = "log_sigmoid_neg"
MODULE = "logsigneg"
CALL = "logsigneg.log_sigmoid_neg(x)"
TOL = 1e-9
SEED = 20260821


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return -np.logaddexp(0.0, x)
