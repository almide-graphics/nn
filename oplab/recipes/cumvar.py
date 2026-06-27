"""Recipe: cumvar_rows (expanding-window population variance over time, src/cumvar.almd)."""
import numpy as np

NAME = "cumvar_rows"
MODULE = "cumvar"
CALL = "cumvar.cumvar_rows(x)"
TOL = 1e-9
SEED = 20260919


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    n = np.arange(1, x.shape[0] + 1).reshape(-1, 1)
    s = np.cumsum(x, axis=0)
    ss = np.cumsum(x * x, axis=0)
    return ss / n - (s / n) ** 2
