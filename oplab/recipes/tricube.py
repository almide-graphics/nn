"""Recipe: tricube (LOWESS kernel (1-|x|^3)_+^3, src/tricube.almd)."""
import numpy as np

NAME = "tricube"
MODULE = "tricube"
CALL = "tricube.tricube(x)"
TOL = 1e-9
SEED = 20260797


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.maximum(0.0, 1.0 - np.abs(x) ** 3) ** 3
