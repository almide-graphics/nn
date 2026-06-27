"""Recipe: kl_div_logits_rows (KL between softmax distributions given as logits, distillation, src/kldivlogits.almd)."""
import numpy as np

NAME = "kl_div_logits_rows"
MODULE = "kldivlogits"
CALL = "kldivlogits.kl_div_logits_rows(a, b)"
TOL = 1e-9
SEED = 20261015


def make_inputs(rng):
    a = rng.standard_normal((5, 4))
    b = rng.standard_normal((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b)}


def _lse(x):
    m = x.max(axis=1, keepdims=True)
    return m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def reference(a, b):
    la = a - _lse(a)
    lb = b - _lse(b)
    p = np.exp(la)
    return (p * (la - lb)).sum(axis=1, keepdims=True)
