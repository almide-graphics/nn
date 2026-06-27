"""Recipe: addcdiv_rows (elementwise fused add-and-divide a + b/c, torch.addcdiv, src/addcdiv.almd)."""
import numpy as np

NAME = "addcdiv_rows"
MODULE = "addcdiv"
CALL = "addcdiv.addcdiv_rows(a, b, c)"
TOL = 1e-9
SEED = 20261056


def make_inputs(rng):
    a = rng.standard_normal((5, 4))
    b = rng.standard_normal((5, 4))
    c = rng.standard_normal((5, 4)) ** 2  # non-negative denominator
    return {"a": ("matrix", a), "b": ("matrix", b), "c": ("matrix", c)}


def reference(a, b, c):
    return a + b / (c + 1e-12)
