"""Recipe: mamba_abar (Mamba S6 state discretization Ā=exp(Δ⊙A), src/mamba.almd)."""
import numpy as np

NAME = "mamba_abar"
MODULE = "mamba"
CALL = "mamba.mamba_abar(delta, a)"
TOL = 1e-9
SEED = 20260711


def make_inputs(rng):
    T, D = 6, 4
    delta = np.abs(rng.standard_normal((T, D))) * 0.5    # Δ > 0 (softplus gate)
    a = -np.abs(rng.standard_normal((T, D)))             # A < 0 (stable)
    return {"delta": ("matrix", delta), "a": ("matrix", a)}


def reference(delta, a):
    return np.exp(delta * a)
