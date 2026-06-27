"""Recipe: relu_cubed (max(0,x)^3, src/relucubed.almd)."""
import numpy as np

NAME = "relu_cubed"
MODULE = "relucubed"
CALL = "relucubed.relu_cubed(x)"
TOL = 1e-9
SEED = 20260827


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.maximum(0.0, x) ** 3
