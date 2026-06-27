"""Recipe: bitnet_b158_quant_error_rows (residual of BitNet b1.58 quantization x − γ·ternary, src/bitnetb158err.almd)."""
import numpy as np

NAME = "bitnet_b158_quant_error_rows"
MODULE = "bitnetb158err"
CALL = "bitnetb158err.bitnet_b158_quant_error_rows(x)"
TOL = 1e-9
SEED = 20260956


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    gamma = np.abs(x).mean()
    u = x / gamma
    code = np.where(u > 0.5, 1.0, np.where(u < -0.5, -1.0, 0.0))
    return x - gamma * code
