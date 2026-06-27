"""Recipe: gaussian_scaled (width-parameterized RBF, src/gaussscaled.almd)."""
import numpy as np

NAME = "gaussian_scaled"
MODULE = "gaussscaled"
CALL = "gaussscaled.gaussian_scaled(x, sigma)"
TOL = 1e-9
SEED = 20260778


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    sigma = float(rng.uniform(0.5, 2.0))
    return {"x": ("matrix", x), "sigma": ("scalar", sigma)}


def reference(x, sigma):
    return np.exp(-x * x / (2.0 * sigma ** 2))
