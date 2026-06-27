"""Recipe: hgrn (hierarchically gated linear RNN, src/hgrn.almd)."""
import numpy as np

NAME = "hgrn"
MODULE = "hgrn"
CALL = "hgrn.hgrn(g, i, lb)"
TOL = 1e-9
SEED = 20260712


def make_inputs(rng):
    T, D = 6, 4
    g = rng.standard_normal((T, D))      # gate logits
    i = rng.standard_normal((T, D))      # candidate input
    lb = 0.9 * rng.random(D)             # per-channel floor in [0, 0.9)
    return {"g": ("matrix", g), "i": ("matrix", i), "lb": ("vector", lb)}


def reference(g, i, lb):
    T, D = g.shape
    h = np.zeros(D)
    out = []
    for t in range(T):
        f = lb + (1.0 - lb) * (1.0 / (1.0 + np.exp(-g[t])))
        h = f * h + (1.0 - f) * i[t]
        out.append(h.copy())
    return np.array(out)
