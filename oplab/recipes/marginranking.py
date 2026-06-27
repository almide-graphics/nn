"""Recipe: margin_ranking_loss_rows (margin/hinge pairwise ranking loss, torch.MarginRankingLoss, src/marginranking.almd)."""
import numpy as np

NAME = "margin_ranking_loss_rows"
MODULE = "marginranking"
CALL = "marginranking.margin_ranking_loss_rows(a, b, y)"
TOL = 1e-9
SEED = 20261027


def make_inputs(rng):
    a = rng.standard_normal((5, 4))
    b = rng.standard_normal((5, 4))
    y = 2.0 * rng.integers(0, 2, size=(5, 4)) - 1.0  # targets in {-1, +1}
    return {"a": ("matrix", a), "b": ("matrix", b), "y": ("matrix", y)}


def reference(a, b, y):
    return np.maximum(0.0, -y * (a - b) + 1.0).mean(axis=1, keepdims=True)
