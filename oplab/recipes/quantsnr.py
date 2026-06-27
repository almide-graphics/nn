"""Recipe: quant_snr_rows (per-row signal-to-quantization-noise ratio of BitNet b1, src/quantsnr.almd)."""
import numpy as np

NAME = "quant_snr_rows"
MODULE = "quantsnr"
CALL = "quantsnr.quant_snr_rows(x)"
TOL = 1e-9
SEED = 20260958


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    beta = np.abs(x).mean()
    sign = np.where(x >= 0.0, 1.0, -1.0)
    e = x - beta * sign
    sig = (x * x).sum(axis=1, keepdims=True)
    noise = (e * e).sum(axis=1, keepdims=True)
    return sig / (noise + 1e-12)
