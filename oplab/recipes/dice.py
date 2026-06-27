"""Recipe: dice_coeff_rows (row-wise Sørensen-Dice coefficient, src/dice.almd)."""
import numpy as np

NAME = "dice_coeff_rows"
MODULE = "dice"
CALL = "dice.dice_coeff_rows(a, b)"
TOL = 1e-9
SEED = 20260990


def make_inputs(rng):
    return {
        "a": ("matrix", rng.standard_normal((5, 4))),
        "b": ("matrix", rng.standard_normal((5, 4))),
    }


def reference(a, b):
    num = 2.0 * (a * b).sum(axis=1, keepdims=True)
    den = (a * a).sum(axis=1, keepdims=True) + (b * b).sum(axis=1, keepdims=True) + 1e-12
    return num / den
