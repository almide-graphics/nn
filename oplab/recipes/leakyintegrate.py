"""Recipe: leaky_integrate_rows (leaky integrator / eligibility trace, src/leakyintegrate.almd)."""
import numpy as np

NAME = "leaky_integrate_rows"
MODULE = "leakyintegrate"
CALL = "leakyintegrate.leaky_integrate_rows(x)"
TOL = 1e-9
SEED = 20261048


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    t, d = x.shape
    out = np.empty((t, d), dtype=float)
    s = x[0].copy()
    out[0] = s
    for i in range(1, t):
        s = x[i] + 0.9 * s
        out[i] = s
    return out
