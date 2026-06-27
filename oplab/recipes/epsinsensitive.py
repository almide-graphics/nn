"""Recipe: eps_insensitive_loss_rows (ε-insensitive / SVR regression loss, ε=0.5, src/epsinsensitive.almd)."""
import numpy as np

NAME = "eps_insensitive_loss_rows"
MODULE = "epsinsensitive"
CALL = "epsinsensitive.eps_insensitive_loss_rows(y, yhat)"
TOL = 1e-9
SEED = 20261036


def make_inputs(rng):
    y = rng.standard_normal((5, 4))
    yhat = rng.standard_normal((5, 4))
    return {"y": ("matrix", y), "yhat": ("matrix", yhat)}


def reference(y, yhat):
    eps = 0.5
    return np.maximum(0.0, np.abs(y - yhat) - eps).mean(axis=1, keepdims=True)
