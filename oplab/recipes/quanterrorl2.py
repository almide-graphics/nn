"""Recipe: quant_error_l2_rows (per-row L2 norm of BitNet b1 quantization residual, src/quanterrorl2.almd)."""
import numpy as np

NAME = "quant_error_l2_rows"
MODULE = "quanterrorl2"
CALL = "quanterrorl2.quant_error_l2_rows(x)"
TOL = 1e-9
SEED = 20260942


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    beta = np.abs(x).mean()
    e = x - beta * np.where(x >= 0.0, 1.0, -1.0)
    return np.sqrt((e * e).sum(axis=1, keepdims=True))
