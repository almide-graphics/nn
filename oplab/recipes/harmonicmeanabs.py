"""Recipe: harmonic_mean_abs_rows (per-row harmonic mean of magnitudes, src/harmonicmeanabs.almd)."""
import numpy as np

NAME = "harmonic_mean_abs_rows"
MODULE = "harmonicmeanabs"
CALL = "harmonicmeanabs.harmonic_mean_abs_rows(x)"
TOL = 1e-9
SEED = 20260979


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    d = x.shape[1]
    inv = (1.0 / (np.abs(x) + 1e-12)).sum(axis=1, keepdims=True)
    return d / inv
