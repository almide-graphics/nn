"""Recipe: ternary_quant_snr_rows (per-row SQNR of TWN ternary quantization, src/ternaryquantsnr.almd)."""
import numpy as np

NAME = "ternary_quant_snr_rows"
MODULE = "ternaryquantsnr"
CALL = "ternaryquantsnr.ternary_quant_snr_rows(x)"
TOL = 1e-9
SEED = 20260973


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    delta = 0.7 * np.abs(x).mean(axis=1, keepdims=True)
    keep = np.abs(x) > delta
    count = keep.sum(axis=1, keepdims=True)
    above = np.where(keep, np.abs(x), 0.0).sum(axis=1, keepdims=True)
    alpha = np.where(count > 0, above / np.where(count > 0, count, 1.0), 0.0)
    q = np.where(x > delta, alpha, np.where(x < -delta, -alpha, 0.0))
    sig = (x * x).sum(axis=1, keepdims=True)
    noise = ((x - q) * (x - q)).sum(axis=1, keepdims=True)
    return sig / (noise + 1e-12)
