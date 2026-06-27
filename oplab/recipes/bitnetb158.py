"""Recipe: bitnet_b158_scaled_rows (BitNet b1.58 dequantized weight γ·ternary, src/bitnetb158.almd)."""
import numpy as np

NAME = "bitnet_b158_scaled_rows"
MODULE = "bitnetb158"
CALL = "bitnetb158.bitnet_b158_scaled_rows(x)"
TOL = 1e-9
SEED = 20260954


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    gamma = np.abs(x).mean()
    u = x / gamma
    code = np.where(u > 0.5, 1.0, np.where(u < -0.5, -1.0, 0.0))
    return gamma * code
