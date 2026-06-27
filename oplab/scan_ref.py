#!/usr/bin/env python3
"""Independent numpy reference oracle for scan_diag (diagonal linear recurrence).

Mirrors the deterministic input formula in oplab/scan.almd bit-for-bit, computes
the reference recurrence, and compares against the Almide-produced output.
This is the op-level analogue of the model-level parity gate in tools/.
"""
import sys
import pathlib
import numpy as np

T, D = 12, 8
ti = np.arange(T)[:, None].astype(np.float64)
j = np.arange(D)[None, :].astype(np.float64)

x = np.sin(0.5 * ti + 0.1 * j)
a = 0.5 + 0.4 * np.sin(0.3 * ti + 0.2 * j)
b = 0.2 + 0.1 * np.sin(0.7 * ti + 0.05 * j)

state = np.zeros(D, dtype=np.float64)
rows = []
for t in range(T):
    state = a[t] * state + b[t] * x[t]
    rows.append(state.copy())
ref = np.array(rows, dtype=np.float64).reshape(-1)

alm_path = pathlib.Path(__file__).with_name("scan_alm.txt")
alm = np.array([float(s) for s in alm_path.read_text().split()], dtype=np.float64)

if alm.size != ref.size:
    print(f"FAIL: size mismatch alm={alm.size} ref={ref.size}")
    sys.exit(1)

abs_err = np.abs(ref - alm)
rel = np.max(abs_err / (np.abs(ref) + 1e-12))
print(f"n={ref.size}  max_abs={abs_err.max():.3e}  max_rel={rel:.3e}")

TOL = 1e-9
ok = rel < TOL
print(("PASS" if ok else "FAIL") + f" (tol rel<{TOL:g})")
sys.exit(0 if ok else 1)
