"""Recipe: reglu (ReLU-gated linear unit, src/reglu.almd)."""
import numpy as np

NAME = "reglu"
MODULE = "reglu"
CALL = "reglu.reglu(x, g)"
TOL = 1e-9
SEED = 20260722


def make_inputs(rng):
    return {
        "x": ("matrix", rng.standard_normal((4, 4))),
        "g": ("matrix", rng.standard_normal((4, 4))),
    }


def reference(x, g):
    return x * np.maximum(g, 0.0)
