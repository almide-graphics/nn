"""Recipe: record_count_rows (running count of record highs over time, record statistics, src/recordcount.almd)."""
import numpy as np

NAME = "record_count_rows"
MODULE = "recordcount"
CALL = "recordcount.record_count_rows(x)"
TOL = 1e-9
SEED = 20261031


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    out = np.empty((t, d), dtype=float)
    best = x[0].copy()
    cnt = np.ones(d)  # the first entry is a record by definition
    for i in range(t):
        upd = x[i] > best
        cnt = cnt + upd.astype(float)
        best = np.where(upd, x[i], best)
        out[i] = cnt
    return out
