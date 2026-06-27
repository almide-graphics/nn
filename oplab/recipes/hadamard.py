"""Recipe: hadamard (elementwise matrix product, src/hadamard.almd)."""
import numpy as np

NAME = "hadamard"
MODULE = "hadamard"
CALL = "hadamard.hadamard(a, b)"
TOL = 1e-9
SEED = 20260721


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((4, 4))),
        "b": ("matrix", rng.standard_normal((4, 4))),
    }


def reference(a, b):
    return a * b
