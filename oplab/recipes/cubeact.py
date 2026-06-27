"""Recipe: cube_act (elementwise x^3, src/cubeact.almd)."""
import numpy as np

NAME = "cube_act"
MODULE = "cubeact"
CALL = "cubeact.cube_act(x)"
TOL = 1e-9
SEED = 20260760


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return x * x * x
