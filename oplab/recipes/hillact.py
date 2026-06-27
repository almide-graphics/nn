"""Recipe: hill_act (Hill gate x^2/(1+x^2), src/hillact.almd)."""
import numpy as np

NAME = "hill_act"
MODULE = "hillact"
CALL = "hillact.hill_act(x)"
TOL = 1e-9
SEED = 20260789


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return x * x / (1.0 + x * x)
