"""Recipe: bitnet_absmean_rows (BitNet b1.58 absmean ternary quantization, src/bitnetabsmean.almd)."""
import numpy as np

NAME = "bitnet_absmean_rows"
MODULE = "bitnetabsmean"
CALL = "bitnetabsmean.bitnet_absmean_rows(x)"
TOL = 1e-9
SEED = 20260923


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    gamma = np.abs(x).mean()
    u = x / gamma
    return np.where(u > 0.5, 1.0, np.where(u < -0.5, -1.0, 0.0))
