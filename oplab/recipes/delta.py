"""Recipe: delta_rule_diag (diagonal DeltaNet update, src/delta.almd)."""
import numpy as np

NAME = "delta_rule_diag"
MODULE = "delta"
CALL = "delta.delta_rule_diag(k, v, beta)"
TOL = 1e-9
SEED = 20260713


def make_inputs(rng):
    T, D = 6, 4
    k = rng.standard_normal((T, D))
    v = rng.standard_normal((T, D))
    beta = rng.random((T, D))            # write strength in [0,1)
    return {"k": ("matrix", k), "v": ("matrix", v), "beta": ("matrix", beta)}


def reference(k, v, beta):
    T, D = k.shape
    s = np.zeros(D)
    out = []
    for t in range(T):
        pred = s * k[t]
        s = s + beta[t] * (v[t] - pred) * k[t]
        out.append(s.copy())
    return np.array(out)
