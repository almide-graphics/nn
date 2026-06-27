"""Recipe: gaussian_nll_rows (per-row Gaussian negative log-likelihood, heteroscedastic, 3-input, src/gaussiannll.almd)."""
import numpy as np

NAME = "gaussian_nll_rows"
MODULE = "gaussiannll"
CALL = "gaussiannll.gaussian_nll_rows(mu, x, sig2)"
TOL = 1e-9
SEED = 20261014


def make_inputs(rng):
    mu = rng.standard_normal((5, 4))
    x = rng.standard_normal((5, 4))
    sig2 = rng.standard_normal((5, 4)) ** 2  # predicted variance is non-negative
    return {"mu": ("matrix", mu), "x": ("matrix", x), "sig2": ("matrix", sig2)}


def reference(mu, x, sig2):
    v = np.maximum(sig2, 1e-6)
    return (0.5 * (np.log(v) + (x - mu) ** 2 / v)).mean(axis=1, keepdims=True)
