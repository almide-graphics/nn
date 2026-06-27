"""Recipe: welsch_loss_rows (per-row Welsch/Leclerc bounded redescending robust loss, src/welsch.almd)."""
import numpy as np

NAME = "welsch_loss_rows"
MODULE = "welsch"
CALL = "welsch.welsch_loss_rows(a, b)"
TOL = 1e-9
SEED = 20261062


def make_inputs(rng):
    a = rng.standard_normal((5, 4))
    b = rng.standard_normal((5, 4))
    return {"a": ("matrix", a), "b": ("matrix", b)}


def reference(a, b):
    r = a - b
    return (0.5 * (1.0 - np.exp(-(r * r)))).mean(axis=1, keepdims=True)
