"""Recipe: ema (per-channel exponential moving average, src/ema.almd)."""
import numpy as np

NAME = "ema"
MODULE = "ema"
CALL = "ema.ema(x, alpha)"
TOL = 1e-9
SEED = 20260714


def make_inputs(rng):
    T, D = 6, 4
    x = rng.standard_normal((T, D))
    alpha = 0.9 * rng.random(D)          # per-channel smoothing in [0, 0.9)
    return {"x": ("matrix", x), "alpha": ("vector", alpha)}


def reference(x, alpha):
    T, D = x.shape
    h = np.zeros(D)
    out = []
    for t in range(T):
        h = alpha * h + (1.0 - alpha) * x[t]
        out.append(h.copy())
    return np.array(out)
