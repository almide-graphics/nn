"""Recipe: cube_sigmoid (sigmoid(x)^3, src/cubesig.almd)."""
import numpy as np

NAME = "cube_sigmoid"
MODULE = "cubesig"
CALL = "cubesig.cube_sigmoid(x)"
TOL = 1e-9
SEED = 20260814


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s ** 3
