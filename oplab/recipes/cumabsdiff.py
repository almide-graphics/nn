"""Recipe: cumsum_abs_diff_rows (cumulative total variation over time, src/cumabsdiff.almd)."""
import numpy as np

NAME = "cumsum_abs_diff_rows"
MODULE = "cumabsdiff"
CALL = "cumabsdiff.cumsum_abs_diff_rows(x)"
TOL = 1e-9
SEED = 20260910


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.cumsum(np.abs(np.diff(x, axis=0, prepend=x[:1])), axis=0)
