"""Recipe: tanh_half (½·tanh(x), src/tanhhalf.almd)."""
import numpy as np

NAME = "tanh_half"
MODULE = "tanhhalf"
CALL = "tanhhalf.tanh_half(x)"
TOL = 1e-9
SEED = 20260858


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return 0.5 * np.tanh(x)
