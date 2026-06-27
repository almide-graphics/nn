"""Recipe: poly1_loss_rows (per-row PolyLoss poly-1, src/poly1loss.almd)."""
import numpy as np

NAME = "poly1_loss_rows"
MODULE = "poly1loss"
CALL = "poly1loss.poly1_loss_rows(logits, target)"
TOL = 1e-9
SEED = 20260991


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
    p = np.exp(logp)
    ce = -(target * logp).sum(axis=1, keepdims=True)
    pt = (target * p).sum(axis=1, keepdims=True)
    return ce + (1.0 - pt)
