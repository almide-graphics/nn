"""Recipe: gla_diag (diagonal gated linear attention, src/gla.almd)."""
import numpy as np

NAME = "gla"
MODULE = "gla"
CALL = "gla.gla_diag(q, k, v, a)"
TOL = 1e-9
SEED = 20260702


def make_inputs(rng):
    T, D = 5, 4
    q = rng.standard_normal((T, D))
    k = rng.standard_normal((T, D))
    v = rng.standard_normal((T, D))
    a = 0.5 + 0.4 * rng.standard_normal((T, D))   # per-channel gate
    return {
        "q": ("matrix", q),
        "k": ("matrix", k),
        "v": ("matrix", v),
        "a": ("matrix", a),
    }


def reference(q, k, v, a):
    T, D = q.shape
    h = np.zeros(D)
    out = []
    for t in range(T):
        h = a[t] * h + k[t] * v[t]
        out.append(q[t] * h)
    return np.array(out)
