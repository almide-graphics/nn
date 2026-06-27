"""Recipe: grad_clip_norm_rows (per-row gradient clipping by L2 norm, max_norm=1, src/gradclipnorm.almd)."""
import numpy as np

NAME = "grad_clip_norm_rows"
MODULE = "gradclipnorm"
CALL = "gradclipnorm.grad_clip_norm_rows(g)"
TOL = 1e-9
SEED = 20261011


def make_inputs(rng):
    return {"g": ("matrix", rng.standard_normal((5, 4)))}


def reference(g):
    n = np.sqrt((g * g).sum(axis=1, keepdims=True))
    scale = np.minimum(1.0, 1.0 / (n + 1e-12))
    return g * scale
