"""Recipe: cumstd_rows (expanding-window population std over time, src/cumstd.almd)."""
import numpy as np

NAME = "cumstd_rows"
MODULE = "cumstd"
CALL = "cumstd.cumstd_rows(x)"
TOL = 1e-9
SEED = 20260927


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    # Same single-pass formula as the op: sqrt(max(E[x^2] - E[x]^2, 0)).
    csum = np.cumsum(x, axis=0)
    csumsq = np.cumsum(x * x, axis=0)
    n = np.arange(1, x.shape[0] + 1).reshape(-1, 1)
    var = csumsq / n - (csum / n) ** 2
    return np.sqrt(np.maximum(var, 0.0))
