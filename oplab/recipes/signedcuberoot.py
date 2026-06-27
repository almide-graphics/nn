"""Recipe: signed_cube_root_rows (signed cube root sign(x)*|x|^(1/3), src/signedcuberoot.almd)."""
import numpy as np

NAME = "signed_cube_root_rows"
MODULE = "signedcuberoot"
CALL = "signedcuberoot.signed_cube_root_rows(x)"
TOL = 1e-9
SEED = 20260941


def make_inputs(rng):
    return {"x": ("matrix", rng.standard_normal((5, 4)))}


def reference(x):
    return np.sign(x) * np.abs(x) ** (1.0 / 3.0)
