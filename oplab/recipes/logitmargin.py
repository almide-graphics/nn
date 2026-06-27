"""Recipe: logit_margin_rows (per-row top1−top2 logit gap, src/logitmargin.almd)."""
import numpy as np

NAME = "logit_margin_rows"
MODULE = "logitmargin"
CALL = "logitmargin.logit_margin_rows(x)"
TOL = 1e-9
SEED = 20260994


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    m1 = x.max(axis=1, keepdims=True)
    masked = np.where(x < m1, x, -1e38)
    m2 = masked.max(axis=1, keepdims=True)
    return m1 - m2
