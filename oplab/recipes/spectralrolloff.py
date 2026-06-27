"""Recipe: spectral_rolloff_rows (per-row spectral roll-off index at 85% cumulative energy, src/spectralrolloff.almd)."""
import numpy as np

NAME = "spectral_rolloff_rows"
MODULE = "spectralrolloff"
CALL = "spectralrolloff.spectral_rolloff_rows(x)"
TOL = 1e-9
SEED = 20261061


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    a = np.abs(x)
    cum = np.cumsum(a, axis=1)            # sequential, matches the op's left-to-right fold
    total = cum[:, -1:]                   # = sequential Σ|x|
    thr = 0.85 * total
    ge = cum >= thr                       # non-decreasing → always True by the last column
    idx = np.argmax(ge, axis=1)          # first crossing index
    return idx.reshape(-1, 1).astype(float)
