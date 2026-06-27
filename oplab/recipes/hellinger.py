"""Recipe: hellinger_dist_rows (Hellinger distance between distributions, src/hellinger.almd)."""
import numpy as np

NAME = "hellinger_dist_rows"
MODULE = "hellinger"
CALL = "hellinger.hellinger_dist_rows(p, q)"
TOL = 1e-9
SEED = 20261028


def make_inputs(rng):
    p = rng.standard_normal((5, 4)) ** 2  # non-negative masses
    q = rng.standard_normal((5, 4)) ** 2
    return {"p": ("matrix", p), "q": ("matrix", q)}


def reference(p, q):
    d = np.sqrt(p) - np.sqrt(q)
    return np.sqrt(0.5 * (d * d).sum(axis=1, keepdims=True))
