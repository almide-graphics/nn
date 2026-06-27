"""Recipe: dirichlet_energy_rows (per-row Dirichlet energy / squared-gradient smoothness, src/dirichletenergy.almd)."""
import numpy as np

NAME = "dirichlet_energy_rows"
MODULE = "dirichletenergy"
CALL = "dirichletenergy.dirichlet_energy_rows(x)"
TOL = 1e-9
SEED = 20261058


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return (np.diff(x, axis=1) ** 2).sum(axis=1, keepdims=True)
