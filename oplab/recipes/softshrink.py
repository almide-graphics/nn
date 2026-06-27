"""Recipe: softshrink (soft thresholding / L1 prox, src/softshrink.almd)."""
import numpy as np

NAME = "softshrink"
MODULE = "softshrink"
CALL = "softshrink.softshrink(x, lambd)"
TOL = 1e-9
SEED = 20260743


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    lambd = float(rng.uniform(0.3, 1.0))
    return {"x": ("matrix", x), "lambd": ("scalar", lambd)}


def reference(x, lambd):
    return np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0.0))
