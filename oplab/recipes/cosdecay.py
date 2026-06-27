"""Recipe: cosine_decay (0.5*(1+cos x), src/cosdecay.almd)."""
import numpy as np

NAME = "cosine_decay"
MODULE = "cosdecay"
CALL = "cosdecay.cosine_decay(x)"
TOL = 1e-9
SEED = 20260779


def make_inputs(rng):
    return {"x": ("matrix", rng.uniform(0.0, np.pi, (4, 4)))}


def reference(x):
    return 0.5 * (1.0 + np.cos(x))
