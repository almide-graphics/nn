"""Recipe: chi_square_dist_rows (symmetric chi-square histogram distance, src/chisquaredist.almd)."""
import numpy as np

NAME = "chi_square_dist_rows"
MODULE = "chisquaredist"
CALL = "chisquaredist.chi_square_dist_rows(a, b)"
TOL = 1e-9
SEED = 20261016


def make_inputs(rng):
    a = rng.standard_normal((5, 4)) ** 2  # non-negative histograms
    b = rng.standard_normal((5, 4)) ** 2
    return {"a": ("matrix", a), "b": ("matrix", b)}


def reference(a, b):
    return 0.5 * ((a - b) ** 2 / (a + b + 1e-12)).sum(axis=1, keepdims=True)
