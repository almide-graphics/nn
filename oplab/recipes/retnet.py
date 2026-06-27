"""Recipe: retnet_retention (RetNet diagonal retention, src/retnet.almd)."""
import numpy as np

NAME = "retnet"
MODULE = "retnet"
CALL = "retnet.retnet_retention(q, k, v, gamma)"
TOL = 1e-9
SEED = 20260710


def make_inputs(rng):
    T, D = 6, 4
    q = rng.standard_normal((T, D))
    k = rng.standard_normal((T, D))
    v = rng.standard_normal((T, D))
    gamma = 0.6 + 0.35 * rng.random(D)   # per-channel decay in [0.6, 0.95)
    return {
        "q": ("matrix", q),
        "k": ("matrix", k),
        "v": ("matrix", v),
        "gamma": ("vector", gamma),
    }


def reference(q, k, v, gamma):
    T, D = q.shape
    s = np.zeros(D)
    out = []
    for t in range(T):
        s = gamma * s + k[t] * v[t]
        out.append(q[t] * s)
    return np.array(out)
