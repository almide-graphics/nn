"""Recipe: hann_window (raised-cosine window, src/hannwindow.almd)."""
import numpy as np

NAME = "hann_window"
MODULE = "hannwindow"
CALL = "hannwindow.hann_window(x)"
TOL = 1e-9
SEED = 20260805


def make_inputs(rng):
    return {"x": ("matrix", rng.uniform(-0.9, 0.9, (4, 4)))}  # interior, off boundary


def reference(x):
    return np.where(np.abs(x) < 1.0, 0.5 * (1.0 + np.cos(np.pi * np.abs(x))), 0.0)
