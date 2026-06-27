"""Recipe: logit (inverse sigmoid log(x/(1-x)), src/logit.almd)."""
import numpy as np

NAME = "logit"
MODULE = "logit"
CALL = "logit.logit(x)"
TOL = 1e-9
SEED = 20260802


def make_inputs(rng):
    return {"x": ("matrix", rng.uniform(0.05, 0.95, (4, 4)))}  # open (0,1)


def reference(x):
    return np.log(x / (1.0 - x))
