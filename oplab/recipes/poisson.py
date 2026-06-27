"""Recipe: poisson_loss_rows (row-wise Poisson NLL, src/poisson.almd)."""
import numpy as np

NAME = "poisson_loss_rows"
MODULE = "poisson"
CALL = "poisson.poisson_loss_rows(p, t)"
TOL = 1e-9
SEED = 20260889


def make_inputs(rng):
    p = np.exp(rng.standard_normal((5, 4)))      # positive rate predictions
    t = np.abs(rng.standard_normal((5, 4)))      # non-negative targets
    return {"p": ("matrix", p), "t": ("matrix", t)}


def reference(p, t):
    return (p - t * np.log(p)).mean(axis=1, keepdims=True)
