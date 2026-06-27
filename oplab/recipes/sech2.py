"""Recipe: sech2 (sech^2(x) = 1 - tanh^2(x), src/sech2.almd)."""
import numpy as np

NAME = "sech2"
MODULE = "sech2"
CALL = "sech2.sech2(x)"
TOL = 1e-9
SEED = 20260792


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((4, 4)))}


def reference(x):
    return 1.0 - np.tanh(x) ** 2
