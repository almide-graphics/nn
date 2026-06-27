"""Recipe: ternary_code_entropy_rows (entropy of TWN ternary code distribution, src/ternarycodeentropy.almd)."""
import numpy as np

NAME = "ternary_code_entropy_rows"
MODULE = "ternarycodeentropy"
CALL = "ternarycodeentropy.ternary_code_entropy_rows(x)"
TOL = 1e-9
SEED = 20260996


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    d = x.shape[1]
    delta = 0.7 * np.abs(x).mean(axis=1, keepdims=True)
    fpos = (x > delta).sum(axis=1, keepdims=True) / d
    fneg = (x < -delta).sum(axis=1, keepdims=True) / d
    fzero = 1.0 - fpos - fneg
    return -(fpos * np.log(fpos + 1e-12) + fneg * np.log(fneg + 1e-12) + fzero * np.log(fzero + 1e-12))
