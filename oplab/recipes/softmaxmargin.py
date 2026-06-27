"""Recipe: softmax_margin_rows (top1−top2 softmax probability / margin sampling, src/softmaxmargin.almd)."""
import numpy as np

NAME = "softmax_margin_rows"
MODULE = "softmaxmargin"
CALL = "softmaxmargin.softmax_margin_rows(x)"
TOL = 1e-9
SEED = 20260995


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    p1 = p.max(axis=1, keepdims=True)
    masked = np.where(p < p1, p, -1e38)
    p2 = masked.max(axis=1, keepdims=True)
    return p1 - p2
