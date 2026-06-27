"""Recipe: pinball_loss_rows (quantile/pinball regression loss at tau=0.7, src/pinball.almd)."""
import numpy as np

NAME = "pinball_loss_rows"
MODULE = "pinball"
CALL = "pinball.pinball_loss_rows(y, yhat)"
TOL = 1e-9
SEED = 20261035


def make_inputs(rng):
    y = rng.standard_normal((5, 4))
    yhat = rng.standard_normal((5, 4))
    return {"y": ("matrix", y), "yhat": ("matrix", yhat)}


def reference(y, yhat):
    tau = 0.7
    e = y - yhat
    return np.maximum(tau * e, (tau - 1.0) * e).mean(axis=1, keepdims=True)
