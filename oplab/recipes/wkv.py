"""Recipe: wkv (RWKV-4 WKV, numerically-stabilized recurrence, src/wkv.almd)."""
import numpy as np

NAME = "wkv"
MODULE = "wkv"
CALL = "wkv.wkv(k, v, w, u)"
TOL = 1e-9
SEED = 20260625


def make_inputs(rng):
    T, D = 5, 3
    k = 0.4 * rng.standard_normal((T, D))
    v = rng.standard_normal((T, D))
    w = np.abs(rng.standard_normal(D)) * 0.6     # per-channel decay (>= 0)
    u = 0.3 * rng.standard_normal(D)             # per-channel bonus
    return {
        "k": ("matrix", k),
        "v": ("matrix", v),
        "w": ("vector", w),
        "u": ("vector", u),
    }


def reference(k, v, w, u):
    T, D = k.shape
    a = np.zeros(D)
    b = np.zeros(D)
    p = np.full(D, -1e38)
    out = []
    for t in range(T):
        q = np.maximum(p, u + k[t])
        e1 = np.exp(p - q)
        e2 = np.exp(u + k[t] - q)
        out.append((e1 * a + e2 * v[t]) / (e1 * b + e2))
        q2 = np.maximum(p - w, k[t])
        f1 = np.exp(p - w - q2)
        f2 = np.exp(k[t] - q2)
        a = f1 * a + f2 * v[t]
        b = f1 * b + f2
        p = q2
    return np.array(out)
