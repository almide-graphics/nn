"""Recipe: relu6_over6 (clip(x,0,6)/6, src/relu6over6.almd)."""
import numpy as np

NAME = "relu6_over6"
MODULE = "relu6over6"
CALL = "relu6over6.relu6_over6(x)"
TOL = 1e-9
SEED = 20260837


def make_inputs(rng):
    return {"x": ("matrix", 3.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.clip(x, 0.0, 6.0) / 6.0
