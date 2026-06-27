"""Recipe: down_run_length_rows (length of current consecutive-decrease streak over time, src/downrunlength.almd)."""
import numpy as np

NAME = "down_run_length_rows"
MODULE = "downrunlength"
CALL = "downrunlength.down_run_length_rows(x)"
TOL = 1e-9
SEED = 20261033


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    out = np.zeros((t, d))
    run = np.zeros(d)
    prev = x[0].copy()
    for i in range(t):
        dn = x[i] < prev
        run = np.where(dn, run + 1.0, 0.0)
        out[i] = run
        prev = x[i]
    return out
