"""Recipe: relu_tanh (max(0, tanh(x)), src/relutanh.almd)."""
import numpy as np

NAME = "relu_tanh"
MODULE = "relutanh"
CALL = "relutanh.relu_tanh(x)"
TOL = 1e-9
SEED = 20260804


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.maximum(0.0, np.tanh(x))
