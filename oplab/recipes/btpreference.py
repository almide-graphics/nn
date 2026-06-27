"""Recipe: bt_preference_rows (Bradley-Terry pairwise preference sigma(a-b), RLHF reward head, src/btpreference.almd)."""
import numpy as np

NAME = "bt_preference_rows"
MODULE = "btpreference"
CALL = "btpreference.bt_preference_rows(a, b)"
TOL = 1e-9
SEED = 20261026


def make_inputs(rng):
    a = rng.standard_normal((5, 4))
    b = rng.standard_normal((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b)}


def reference(a, b):
    return 1.0 / (1.0 + np.exp(-(a - b)))
