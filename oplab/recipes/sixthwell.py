"""Recipe: sixth_well (degree-6 well (x^2-1)^3, src/sixthwell.almd)."""
import numpy as np

NAME = "sixth_well"
MODULE = "sixthwell"
CALL = "sixthwell.sixth_well(x)"
TOL = 1e-9
SEED = 20260864


def make_inputs(rng):
    return {"x": ("matrix", 1.4 * rng.standard_normal((4, 4)))}


def reference(x):
    return (x * x - 1.0) ** 3
