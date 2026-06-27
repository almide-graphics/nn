"""Recipe: snake (x + sin^2(x), src/snake.almd)."""
import numpy as np

NAME = "snake"
MODULE = "snake"
CALL = "snake.snake(x)"
TOL = 1e-9
SEED = 20260773


def make_inputs(rng):
    return {"x": ("matrix", 2.0 * rng.standard_normal((4, 4)))}


def reference(x):
    return x + np.sin(x) ** 2
