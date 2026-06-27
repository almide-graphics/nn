"""Recipe: sign_match_rate_rows (per-row fraction of sign agreements, src/signmatchrate.almd)."""
import numpy as np

NAME = "sign_match_rate_rows"
MODULE = "signmatchrate"
CALL = "signmatchrate.sign_match_rate_rows(a, b)"
TOL = 1e-9
SEED = 20260943


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    return (np.sign(a) == np.sign(b)).mean(axis=1, keepdims=True)
