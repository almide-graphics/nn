"""Recipe: bce_rows (row-wise binary cross-entropy, src/bce.almd)."""
import numpy as np

NAME = "bce_rows"
MODULE = "bce"
CALL = "bce.bce_rows(p, t)"
TOL = 1e-9
SEED = 20260884


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_inputs(rng):
    # p, t in (0,1) so log(p), log(1-p) are finite
    p = _sigmoid(rng.standard_normal((5, 4)))
    t = _sigmoid(rng.standard_normal((5, 4)))
    return {"p": ("matrix", p), "t": ("matrix", t)}


def reference(p, t):
    contrib = -(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))
    return contrib.mean(axis=1, keepdims=True)
