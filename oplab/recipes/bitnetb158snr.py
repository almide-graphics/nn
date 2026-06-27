"""Recipe: bitnet_b158_quant_snr_rows (per-row SQNR of BitNet b1.58 quantization, src/bitnetb158snr.almd)."""
import numpy as np

NAME = "bitnet_b158_quant_snr_rows"
MODULE = "bitnetb158snr"
CALL = "bitnetb158snr.bitnet_b158_quant_snr_rows(x)"
TOL = 1e-9
SEED = 20260975


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    gamma = np.abs(x).mean()
    u = x / gamma
    code = np.where(u > 0.5, 1.0, np.where(u < -0.5, -1.0, 0.0))
    q = gamma * code
    sig = (x * x).sum(axis=1, keepdims=True)
    noise = ((x - q) * (x - q)).sum(axis=1, keepdims=True)
    return sig / (noise + 1e-12)
