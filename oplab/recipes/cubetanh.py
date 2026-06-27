"""Recipe: cube_tanh (tanh(x)^3, src/cubetanh.almd)."""
import numpy as np

NAME = "cube_tanh"
MODULE = "cubetanh"
CALL = "cubetanh.cube_tanh(x)"
TOL = 1e-9
SEED = 20260865


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.tanh(x) ** 3
