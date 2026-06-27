"""Recipe: sign_entropy_rows (binary entropy of each row's sign distribution, src/signentropy.almd)."""
import numpy as np

NAME = "sign_entropy_rows"
MODULE = "signentropy"
CALL = "signentropy.sign_entropy_rows(x)"
TOL = 1e-9
SEED = 20260978


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    d = x.shape[1]
    p = (x > 0).sum(axis=1, keepdims=True) / d
    q = 1.0 - p
    return -p * np.log(p + 1e-12) - q * np.log(q + 1e-12)
