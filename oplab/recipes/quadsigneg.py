"""Recipe: quad_times_sigmoid_neg (x^2 * sigmoid(-x), src/quadsigneg.almd)."""
import numpy as np

NAME = "quad_times_sigmoid_neg"
MODULE = "quadsigneg"
CALL = "quadsigneg.quad_times_sigmoid_neg(x)"
TOL = 1e-9
SEED = 20260853


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return x * x / (1.0 + np.exp(x))
