"""Recipe: scan_diag (diagonal linear recurrence, src/scan.almd)."""
import numpy as np

NAME = "scan_diag"
MODULE = "scan"
CALL = "scan.scan_diag(a, b, x)"
TOL = 1e-9
SEED = 20260624


def make_inputs(rng):
    T, D = 6, 4
    a = 0.5 + 0.4 * rng.standard_normal((T, D))   # decay, mixed sign on purpose
    b = rng.standard_normal((T, D))               # input gate
    x = rng.standard_normal((T, D))               # input
    return {"a": ("matrix", a), "b": ("matrix", b), "x": ("matrix", x)}


def reference(a, b, x):
    T, D = x.shape
    state = np.zeros(D)
    out = []
    for t in range(T):
        state = a[t] * state + b[t] * x[t]
        out.append(state.copy())
    return np.array(out)
