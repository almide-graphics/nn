"""Recipe: morlet_approx (Morlet wavelet cos(5x)exp(-x^2/2), src/morlet.almd)."""
import numpy as np

NAME = "morlet_approx"
MODULE = "morlet"
CALL = "morlet.morlet_approx(x)"
TOL = 1e-9
SEED = 20260801


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return np.cos(5.0 * x) * np.exp(-x * x / 2.0)
