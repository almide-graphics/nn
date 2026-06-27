"""Recipe: logcosh_loss_rows (row-wise log-cosh loss, src/logcoshloss.almd)."""
import numpy as np

NAME = "logcosh_loss_rows"
MODULE = "logcoshloss"
CALL = "logcoshloss.logcosh_loss_rows(a, b)"
TOL = 1e-9
SEED = 20260888


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    ea = np.abs(a - b)
    lc = ea + np.log(1.0 + np.exp(-2.0 * ea)) - np.log(2.0)
    return lc.mean(axis=1, keepdims=True)
