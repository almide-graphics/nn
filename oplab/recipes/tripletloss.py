"""Recipe: triplet_loss_rows (per-row triplet margin loss, 3-input, src/tripletloss.almd)."""
import numpy as np

NAME = "triplet_loss_rows"
MODULE = "tripletloss"
CALL = "tripletloss.triplet_loss_rows(a, p, n)"
TOL = 1e-9
SEED = 20261004


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "p": ("matrix", rng.standard_normal((5, 4))),
        "n": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, p, n):
    dap = np.sqrt(((a - p) * (a - p)).sum(axis=1, keepdims=True))
    dan = np.sqrt(((a - n) * (a - n)).sum(axis=1, keepdims=True))
    return np.maximum(0.0, dap - dan + 1.0)
