#!/usr/bin/env python3
"""HF greedy reference for the E2E soft check (L1-5).

Prints the prompt IDs, the greedy continuation IDs, and the decoded text
so the Almide side (examples/qwen_chat.almd) can be diffed by eye or
script.

Usage:
    tools/.venv/bin/python tools/greedy_ref.py --prompt-file parity/token_ids_0.txt --max-new 16
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-file", default="parity/token_ids_0.txt")
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    ids = [int(x) for x in Path(args.prompt_file).read_text().split(",")]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    out = model.generate(
        torch.tensor([ids]),
        do_sample=False,
        max_new_tokens=args.max_new,
        pad_token_id=tok.eos_token_id,
    )[0].tolist()

    new = out[len(ids):]
    print("prompt ids:", ",".join(map(str, ids)))
    print("greedy ids:", ",".join(map(str, new)))
    print("text:", repr(tok.decode(out)))


if __name__ == "__main__":
    main()
