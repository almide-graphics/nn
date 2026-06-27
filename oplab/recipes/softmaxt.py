"""Recipe: softmax_temperature (row-wise temperature softmax, src/softmaxt.almd)."""
import numpy as np

NAME = "softmax_temperature"
MODULE = "softmaxt"
CALL = "softmaxt.softmax_temperature(x, tau)"
TOL = 1e-9
SEED = 20260715


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 5))
    tau = float(rng.uniform(0.5, 2.0))   # scalar temperature
    return {"x": ("matrix", x), "tau": ("scalar", tau)}


def reference(x, tau):
    z = x / tau
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)
