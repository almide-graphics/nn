"""Recipe: cross_entropy_rows (per-row softmax cross-entropy, src/celoss.almd)."""
import numpy as np

NAME = "cross_entropy_rows"
MODULE = "celoss"
CALL = "celoss.cross_entropy_rows(logits, target)"
TOL = 1e-9
SEED = 20260720


def make_inputs(rng):
    T, D = 5, 4
    logits = 2.0 * rng.standard_normal((T, D))
    idx = rng.integers(0, D, size=T)
    target = np.zeros((T, D))
    target[np.arange(T), idx] = 1.0
    return {"logits": ("matrix", logits), "target": ("matrix", target)}


def reference(logits, target):
    m = logits.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    logp = logits - lse
    return -(target * logp).sum(axis=1, keepdims=True)
