"""Recipe: lerp_rows (elementwise linear interpolation a + t·(b−a), 3-input, src/lerp.almd)."""
import numpy as np

NAME = "lerp_rows"
MODULE = "lerp"
CALL = "lerp.lerp_rows(a, b, t)"
TOL = 1e-9
SEED = 20261005


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
        "t": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b, t):
    return a + t * (b - a)
