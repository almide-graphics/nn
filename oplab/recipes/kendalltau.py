"""Recipe: kendall_tau_rows (per-row Kendall rank correlation, all-pairs concordance, src/kendalltau.almd)."""
import numpy as np

NAME = "kendall_tau_rows"
MODULE = "kendalltau"
CALL = "kendalltau.kendall_tau_rows(a, b)"
TOL = 1e-9
SEED = 20261064


def make_inputs(rng):
    a = rng.standard_normal((5, 4))
    b = rng.standard_normal((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b)}


def reference(a, b):
    t, d = a.shape
    npairs = d * (d - 1) / 2
    out = np.zeros((t, 1))
    for i in range(t):
        s = 0.0
        for j in range(d):
            for k in range(j + 1, d):
                p = (a[i, j] - a[i, k]) * (b[i, j] - b[i, k])
                s += np.sign(p)
        out[i, 0] = s / npairs
    return out
