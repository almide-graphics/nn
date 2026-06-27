"""Recipe: silu (SiLU/swish activation, src/silu.almd). Non-recurrence op —
shows the flywheel verifier is op-shape agnostic."""
import numpy as np

NAME = "silu"
MODULE = "silu"
CALL = "silu.silu(x)"
TOL = 1e-9
SEED = 20260626


def make_inputs(rng):
    x = 1.5 * rng.standard_normal((4, 3))   # |x| stays well below exp-overflow
    return {"x": ("matrix", x)}


def reference(x):
    return x / (1.0 + np.exp(-x))
