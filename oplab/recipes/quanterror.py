"""Recipe: quant_error_rows (residual of BitNet b1 1-bit quantization, src/quanterror.almd)."""
import numpy as np

NAME = "quant_error_rows"
MODULE = "quanterror"
CALL = "quanterror.quant_error_rows(x)"
TOL = 1e-9
SEED = 20260935


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    beta = np.abs(x).mean()
    return x - beta * np.where(x >= 0.0, 1.0, -1.0)
