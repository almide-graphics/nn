"""Recipe: clip_symmetric (clamp to [-c,c], src/clipsym.almd)."""
import numpy as np

NAME = "clip_symmetric"
MODULE = "clipsym"
CALL = "clipsym.clip_symmetric(x, c)"
TOL = 1e-9
SEED = 20260775


def make_inputs(rng):
    x = 2.0 * rng.standard_normal((4, 4))
    c = float(rng.uniform(0.5, 2.0))
    return {"x": ("matrix", x), "c": ("scalar", c)}


def reference(x, c):
    return np.clip(x, -c, c)
