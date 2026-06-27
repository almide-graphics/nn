"""Recipe: bitnet_quant_scaled_rows (BitNet b1 binary weight times absmean dequant scale, src/bitnetquantscaled.almd)."""
import numpy as np

NAME = "bitnet_quant_scaled_rows"
MODULE = "bitnetquantscaled"
CALL = "bitnetquantscaled.bitnet_quant_scaled_rows(x)"
TOL = 1e-9
SEED = 20260928


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    # Per-tensor absmean scale beta, then dequantized 1-bit sign (zero -> +1).
    beta = np.abs(x).mean()
    return beta * np.where(x >= 0.0, 1.0, -1.0)
