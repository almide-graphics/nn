"""Recipe: rank_rows (per-row ordinal rank transform via all-pairs count, sort-free, src/rankrows.almd)."""
import numpy as np

NAME = "rank_rows"
MODULE = "rankrows"
CALL = "rankrows.rank_rows(x)"
TOL = 1e-9
SEED = 20261063


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    # rank[i,j] = #{k : x[i,k] < x[i,j]}
    gt = x[:, :, None] > x[:, None, :]   # gt[i,j,k] = x[i,j] > x[i,k] = x[i,k] < x[i,j]
    return gt.sum(axis=2).astype(float)
