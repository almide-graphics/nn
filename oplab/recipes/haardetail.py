"""Recipe: haar_detail_x2_rows (level-1 Haar wavelet detail / high-pass, downsampling, src/haardetail.almd)."""
import numpy as np

NAME = "haar_detail_x2_rows"
MODULE = "haardetail"
CALL = "haardetail.haar_detail_x2_rows(x)"
TOL = 1e-9
SEED = 20261041


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return (x[:, 0::2] - x[:, 1::2]) / 2.0
