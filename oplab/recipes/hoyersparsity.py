"""Recipe: hoyer_sparsity_rows (per-row Hoyer sparsity, L1/L2 ratio, src/hoyersparsity.almd)."""
import numpy as np

NAME = "hoyer_sparsity_rows"
MODULE = "hoyersparsity"
CALL = "hoyersparsity.hoyer_sparsity_rows(x)"
TOL = 1e-9
SEED = 20260983


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    d = x.shape[1]
    l1 = np.abs(x).sum(axis=1, keepdims=True)
    l2 = np.sqrt((x * x).sum(axis=1, keepdims=True))
    sq = np.sqrt(d)
    return (sq - l1 / (l2 + 1e-12)) / (sq - 1.0)
