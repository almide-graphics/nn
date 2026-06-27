"""Recipe: up_run_length_rows (length of current consecutive-increase streak over time, src/uprunlength.almd)."""
import numpy as np

NAME = "up_run_length_rows"
MODULE = "uprunlength"
CALL = "uprunlength.up_run_length_rows(x)"
TOL = 1e-9
SEED = 20261032


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    out = np.zeros((t, d))
    run = np.zeros(d)
    prev = x[0].copy()
    for i in range(t):
        up = x[i] > prev
        run = np.where(up, run + 1.0, 0.0)
        out[i] = run
        prev = x[i]
    return out
