"""Recipe: signed_log1p_rows (sign-preserving log compression, src/signedlog1p.almd)."""
import numpy as np

NAME = "signed_log1p_rows"
MODULE = "signedlog1p"
CALL = "signedlog1p.signed_log1p_rows(x)"
TOL = 1e-9
SEED = 20260937


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    # log(1+|x|) to match the op exactly (not log1p).
    return np.sign(x) * np.log(1.0 + np.abs(x))
