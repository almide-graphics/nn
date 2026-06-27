"""Recipe: relu6 (ReLU clamped at 6, src/relu6.almd)."""
import numpy as np

NAME = "relu6"
MODULE = "relu6"
CALL = "relu6.relu6(x)"
TOL = 1e-9
SEED = 20260718


def make_inputs(rng):
    x = 4.0 * rng.standard_normal((4, 4))   # cover <0, [0,6], >6
    return {"x": ("matrix", x)}


def reference(x):
    return np.clip(x, 0.0, 6.0)
