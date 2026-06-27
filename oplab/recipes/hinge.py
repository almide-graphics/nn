"""Recipe: hinge_rows (row-wise SVM hinge loss, src/hinge.almd)."""
import numpy as np

NAME = "hinge_rows"
MODULE = "hinge"
CALL = "hinge.hinge_rows(t, s)"
TOL = 1e-9
SEED = 20260885


def make_inputs(rng):
    t = np.sign(rng.standard_normal((5, 4)))  # ±1 targets (randn ~never exactly 0)
    s = rng.standard_normal((5, 4))           # raw scores
    return {"t": ("matrix", t), "s": ("matrix", s)}


def reference(t, s):
    return np.maximum(0.0, 1.0 - t * s).mean(axis=1, keepdims=True)
