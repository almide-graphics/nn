"""Recipe: bitnet_sign_rows (BitNet b1 per-element binarization, src/bitnetsign.almd)."""
import numpy as np

NAME = "bitnet_sign_rows"
MODULE = "bitnetsign"
CALL = "bitnetsign.bitnet_sign_rows(x)"
TOL = 1e-9
SEED = 20260922


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.where(x >= 0, 1.0, -1.0)
