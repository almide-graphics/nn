"""Recipe: sinc (sin(x)/x, sinc(0)=1, src/sinc.almd)."""
import numpy as np

NAME = "sinc"
MODULE = "sinc"
CALL = "sinc.sinc(x)"
TOL = 1e-9
SEED = 20260803


def make_inputs(rng):
    sign = np.where(rng.standard_normal((4, 4)) > 0, 1.0, -1.0)
    mag = rng.uniform(0.3, 5.0, (4, 4))      # keep |x| away from 0
    return {"x": ("matrix", sign * mag)}


def reference(x):
    a = np.abs(x)
    return np.where(a < 1e-7, 1.0, np.sin(a) / a)   # mirror Almide's sin(|x|)/|x|
