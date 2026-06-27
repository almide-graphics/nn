"""Recipe: entropy_rows (per-row Shannon entropy, src/entropy.almd)."""
import numpy as np

NAME = "entropy_rows"
MODULE = "entropy"
CALL = "entropy.entropy_rows(p)"
TOL = 1e-9
SEED = 20260725


def make_inputs(rng):
    z = rng.standard_normal((5, 4))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)     # valid probability rows
    return {"p": ("matrix", p)}


def reference(p):
    return -(p * np.log(p + 1e-12)).sum(axis=1, keepdims=True)
