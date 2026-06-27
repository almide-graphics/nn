"""Recipe: cauchy_act (Lorentzian bump 1/(1+x^2), src/cauchyact.almd)."""
import numpy as np

NAME = "cauchy_act"
MODULE = "cauchyact"
CALL = "cauchyact.cauchy_act(x)"
TOL = 1e-9
SEED = 20260786


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return 1.0 / (1.0 + x * x)
