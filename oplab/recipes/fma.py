"""Recipe: fma_rows (elementwise fused multiply-add a*b + c, 3-input, src/fma.almd)."""
import numpy as np

NAME = "fma_rows"
MODULE = "fma"
CALL = "fma.fma_rows(a, b, c)"
TOL = 1e-9
SEED = 20261006


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
        "c": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b, c):
    return a * b + c
