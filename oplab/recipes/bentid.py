"""Recipe: bent_identity ((sqrt(x^2+1)-1)/2 + x, src/bentid.almd)."""
import numpy as np

NAME = "bent_identity"
MODULE = "bentid"
CALL = "bentid.bent_identity(x)"
TOL = 1e-9
SEED = 20260784


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return (np.sqrt(x * x + 1.0) - 1.0) / 2.0 + x
