#!/usr/bin/env python3
"""Dump reference logits from HF Qwen3 (fp32, CPU) for the Almide parity check.

Writes to the output dir, per prompt i:
    token_ids_{i}.txt   comma-separated input token IDs (Almide side input)
    ref_logits_{i}.bin  float32 LE logits of the LAST position (vocab,)
    ref_top1_{i}.txt    comma-separated argmax token ID per position
plus prompts.json (manifest with the prompt texts).

Usage:
    tools/.venv/bin/python tools/dump_logits.py --out parity
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "def fibonacci(n):",
    "Once upon a time, there was a",
    "The quick brown fox jumps over the lazy dog.",
    "Q: What is the speed of light?\nA:",
    "import numpy as np\n",
    "こんにちは。今日は",
    "日本で一番高い山は",
    "人工知能とは、",
    "メロスは激怒した。",
    "東京の天気は",
    "SELECT * FROM users WHERE",
    "fn main() {",
    "# How to install\n\nRun the following command:",
    "3.14159 is the value of",
    "A",
    "The answer to life, the universe, and everything is",
    "猫が好きです。犬も",
    "Translate to English: 吾輩は猫である。",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="parity")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    manifest = []
    for i, text in enumerate(PROMPTS):
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits[0]  # (seq, vocab) fp32

        (out / f"token_ids_{i}.txt").write_text(
            ",".join(str(t) for t in ids[0].tolist())
        )
        np.asarray(logits[-1].numpy(), dtype="<f4").tofile(out / f"ref_logits_{i}.bin")
        (out / f"ref_top1_{i}.txt").write_text(
            ",".join(str(t) for t in logits.argmax(-1).tolist())
        )
        manifest.append({"id": i, "text": text, "n_tokens": ids.shape[1]})
        print(f"[{i:2d}] {ids.shape[1]:3d} tokens  {text[:40]!r}")

    (out / "prompts.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1)
    )
    print(f"wrote {len(PROMPTS)} prompts to {out}/")


if __name__ == "__main__":
    main()
