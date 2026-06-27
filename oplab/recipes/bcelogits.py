"""Recipe: bce_with_logits_rows (numerically stable BCE from logits, src/bcelogits.almd)."""
import numpy as np

NAME = "bce_with_logits_rows"
MODULE = "bcelogits"
CALL = "bcelogits.bce_with_logits_rows(z, y)"
TOL = 1e-9
SEED = 20261013


def make_inputs(rng):
    z = rng.standard_normal((5, 4))
    y = rng.integers(0, 2, size=(5, 4)).astype(float)
    return {"z": ("matrix", z), "y": ("matrix", y)}


def reference(z, y):
    loss = np.maximum(z, 0.0) - z * y + np.log(1.0 + np.exp(-np.abs(z)))
    return loss.mean(axis=1, keepdims=True)
