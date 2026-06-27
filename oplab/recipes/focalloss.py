"""Recipe: focal_loss_rows (per-row softmax focal loss, γ=2, src/focalloss.almd)."""
import numpy as np

NAME = "focal_loss_rows"
MODULE = "focalloss"
CALL = "focalloss.focal_loss_rows(logits, target)"
TOL = 1e-9
SEED = 20260989


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
    focal = target * (1.0 - p) * (1.0 - p) * logp
    return -focal.sum(axis=1, keepdims=True)
