"""Recipe: exp_neg_abs_cubed_half (gentle super-Gaussian exp(-|x|^3/2), src/expnegabscubedhalf.almd)."""
import numpy as np

NAME = "exp_neg_abs_cubed_half"
MODULE = "expnegabscubedhalf"
CALL = "expnegabscubedhalf.exp_neg_abs_cubed_half(x)"
TOL = 1e-9
SEED = 20260863


def make_inputs(rng):
    return {"x": ("matrix", 1.3 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.exp(-0.5 * np.abs(x) ** 3)
