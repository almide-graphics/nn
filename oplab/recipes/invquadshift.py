"""Recipe: inv_quad_shift (Lorentzian at 1, 1/(1+(x-1)^2), src/invquadshift.almd)."""
import numpy as np

NAME = "inv_quad_shift"
MODULE = "invquadshift"
CALL = "invquadshift.inv_quad_shift(x)"
TOL = 1e-9
SEED = 20260844


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return 1.0 / (1.0 + (x - 1.0) ** 2)
