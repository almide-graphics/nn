"""Recipe: bipolar_sigmoid (2*sigmoid(x)-1 = tanh(x/2), src/bipolarsig.almd)."""
import numpy as np

NAME = "bipolar_sigmoid"
MODULE = "bipolarsig"
CALL = "bipolarsig.bipolar_sigmoid(x)"
TOL = 1e-9
SEED = 20260746


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return 2.0 / (1.0 + np.exp(-x)) - 1.0
