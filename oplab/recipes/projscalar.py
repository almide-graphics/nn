"""Recipe: proj_scalar_rows (scalar projection of a onto b, src/projscalar.almd)."""
import numpy as np

NAME = "proj_scalar_rows"
MODULE = "projscalar"
CALL = "projscalar.proj_scalar_rows(a, b)"
TOL = 1e-9
SEED = 20260997


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    dot = (a * b).sum(axis=1, keepdims=True)
    nb = np.sqrt((b * b).sum(axis=1, keepdims=True))
    return dot / (nb + 1e-12)
