"""Recipe: quartic_well (double-well (x^2-1)^2, src/quarticwell.almd)."""
import numpy as np

NAME = "quartic_well"
MODULE = "quarticwell"
CALL = "quarticwell.quartic_well(x)"
TOL = 1e-9
SEED = 20260862


def make_inputs(rng):
    return {"x": ("matrix", 1.5 * rng.standard_normal((4, 4)))}


def reference(x):
    return (x * x - 1.0) ** 2
