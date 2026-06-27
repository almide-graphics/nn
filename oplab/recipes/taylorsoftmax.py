"""Recipe: taylor_softmax_rows (2nd-order Taylor polynomial softmax over each row, src/taylorsoftmax.almd)."""
import numpy as np

NAME = "taylor_softmax_rows"
MODULE = "taylorsoftmax"
CALL = "taylorsoftmax.taylor_softmax_rows(x)"
TOL = 1e-9
SEED = 20260986


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    num = 1.0 + x + 0.5 * x * x
    return num / num.sum(axis=1, keepdims=True)
