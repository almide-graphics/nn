"""Recipe: bitnet_b158_density_rows (per-row kept-weight density under BitNet b1.58 per-tensor cut, src/bitnetb158density.almd)."""
import numpy as np

NAME = "bitnet_b158_density_rows"
MODULE = "bitnetb158density"
CALL = "bitnetb158density.bitnet_b158_density_rows(x)"
TOL = 1e-9
SEED = 20260971


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    gamma = np.abs(x).mean()
    keep = (np.abs(x) > 0.5 * gamma).astype(float)
    return keep.mean(axis=1, keepdims=True)
