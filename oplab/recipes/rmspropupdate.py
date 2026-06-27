"""Recipe: rmsprop_update_rows (preconditioned gradient g/(sqrt(v)+eps), RMSProp/Adam step, src/rmspropupdate.almd)."""
import numpy as np

NAME = "rmsprop_update_rows"
MODULE = "rmspropupdate"
CALL = "rmspropupdate.rmsprop_update_rows(g, v)"
TOL = 1e-9
SEED = 20261010


def make_inputs(rng):
    g = rng.standard_normal((5, 4))
    v = rng.standard_normal((5, 4)) ** 2  # second moment is non-negative
    return {"g": ("matrix", g), "v": ("matrix", v)}


def reference(g, v):
    return g / (np.sqrt(v) + 1e-8)
