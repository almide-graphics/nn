"""Recipe: ema_sq_rows (exponential moving average of squares, RMSProp/Adam v-accumulator, src/emasq.almd)."""
import numpy as np

NAME = "ema_sq_rows"
MODULE = "emasq"
CALL = "emasq.ema_sq_rows(x)"
TOL = 1e-9
SEED = 20261009


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    beta = 0.9
    x2 = x * x
    out = np.empty_like(x, dtype=float)
    s = np.zeros(x.shape[1])
    for i in range(x.shape[0]):
        s = beta * s + (1.0 - beta) * x2[i]
        out[i] = s
    return out
