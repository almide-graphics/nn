"""Recipe: power_act (|x|^1.5, src/powact.almd)."""
import numpy as np

NAME = "power_act"
MODULE = "powact"
CALL = "powact.power_act(x)"
TOL = 1e-9
SEED = 20260772


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.abs(x) ** 1.5
