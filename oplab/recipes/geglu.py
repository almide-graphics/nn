"""Recipe: geglu (GELU-gated linear unit, src/geglu.almd)."""
import numpy as np

NAME = "geglu"
MODULE = "geglu"
CALL = "geglu.geglu(x, g)"
TOL = 1e-9
SEED = 20260736


def make_inputs(rng):
    return {
        "x": ("matrix", rng.standard_normal((4, 4))),
        "g": ("matrix", rng.standard_normal((4, 4))),
    }


def reference(x, g):
    gelu = 0.5 * g * (1.0 + np.tanh(0.7978845608028654 * (g + 0.044715 * g**3)))
    return x * gelu
