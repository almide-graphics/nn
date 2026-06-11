#!/usr/bin/env python3
"""Compare Almide logits against the HF reference dump.

Expects in the parity dir, per prompt i:
    ref_logits_{i}.bin / ref_top1_{i}.txt   (from dump_logits.py)
    alm_logits_{i}.bin / alm_top1_{i}.txt   (from examples/_parity_qwen3.almd)

Pass criteria (L1 gate):
    last-position argmax matches on every prompt
    per-position top-1 agreement >= 99%
    max relative error of last-position logits < 1e-3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="parity")
    args = ap.parse_args()
    d = Path(args.dir)

    manifest = json.loads((d / "prompts.json").read_text())
    n_pass = 0
    total_pos = 0
    match_pos = 0
    worst_rel = 0.0
    failed = []

    print(f"{'id':>3} {'argmax':>7} {'pos-top1':>9} {'max-rel':>10}  text")
    for entry in manifest:
        i = entry["id"]
        alm_txt = d / f"alm_logits_{i}.txt"
        if not alm_txt.exists():
            print(f"{i:3d} {'MISSING':>7}")
            failed.append(i)
            continue
        ref = np.fromfile(d / f"ref_logits_{i}.bin", dtype="<f4").astype(np.float64)
        alm = np.array([float(x) for x in alm_txt.read_text().split(",")])
        if ref.shape != alm.shape:
            print(f"{i:3d} shape mismatch ref={ref.shape} alm={alm.shape}")
            failed.append(i)
            continue

        argmax_ok = int(ref.argmax()) == int(alm.argmax())
        scale = np.abs(ref).max()
        rel = float(np.abs(ref - alm).max() / (scale + 1e-12))
        worst_rel = max(worst_rel, rel)

        ref_top1 = [int(x) for x in (d / f"ref_top1_{i}.txt").read_text().split(",")]
        alm_top1 = [int(x) for x in (d / f"alm_top1_{i}.txt").read_text().split(",")]
        agree = sum(a == b for a, b in zip(ref_top1, alm_top1))
        total_pos += len(ref_top1)
        match_pos += agree

        ok = argmax_ok and rel < 1e-3
        n_pass += ok
        if not ok:
            failed.append(i)
        print(
            f"{i:3d} {'ok' if argmax_ok else 'FAIL':>7} "
            f"{agree:4d}/{len(ref_top1):<4d} {rel:10.2e}  {entry['text'][:36]!r}"
        )

    pos_rate = match_pos / total_pos if total_pos else 0.0
    print(
        f"\n{n_pass}/{len(manifest)} prompts pass, "
        f"per-position top-1 {match_pos}/{total_pos} ({pos_rate:.2%}), "
        f"worst rel err {worst_rel:.2e}"
    )
    gate = n_pass == len(manifest) and pos_rate >= 0.99
    print("L1 GATE:", "PASS" if gate else "FAIL")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
