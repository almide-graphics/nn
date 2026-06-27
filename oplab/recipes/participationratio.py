"""Recipe: participation_ratio_rows (per-row participation ratio / effective dimensionality, src/participationratio.almd)."""
import numpy as np

NAME = "participation_ratio_rows"
MODULE = "participationratio"
CALL = "participationratio.participation_ratio_rows(x)"
TOL = 1e-9
SEED = 20261000


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    x2 = x * x
    s2 = x2.sum(axis=1, keepdims=True)
    s4 = (x2 * x2).sum(axis=1, keepdims=True)
    return s2 * s2 / (s4 + 1e-12)
