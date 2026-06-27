"""Recipe: attention_pool_rows (softmax-weighted pooling of values by scores, src/attentionpool.almd)."""
import numpy as np

NAME = "attention_pool_rows"
MODULE = "attentionpool"
CALL = "attentionpool.attention_pool_rows(x, w)"
TOL = 1e-9
SEED = 20260984


def make_inputs(rng):
    return {
        "x": ("matrix", rng.standard_normal((5, 4))),
        "w": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(x, w):
    e = np.exp(w - w.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return (p * x).sum(axis=1, keepdims=True)
