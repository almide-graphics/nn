"""Recipe: softmax_collision_entropy_rows (Rényi-2 collision entropy of softmax(logits), src/softmaxcollision.almd)."""
import numpy as np

NAME = "softmax_collision_entropy_rows"
MODULE = "softmaxcollision"
CALL = "softmaxcollision.softmax_collision_entropy_rows(x)"
TOL = 1e-9
SEED = 20260970


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return -np.log((p * p).sum(axis=1, keepdims=True) + 1e-12)
