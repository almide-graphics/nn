"""Recipe: argmax_rows (per-row hard argmax index, src/argmax.almd)."""
import numpy as np

NAME = "argmax_rows"
MODULE = "argmax"
CALL = "argmax.argmax_rows(x)"
TOL = 1e-9
SEED = 20261049


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.argmax(x, axis=1, keepdims=True).astype(float)
