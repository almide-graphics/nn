#!/usr/bin/env python3
"""Export a tiny seeded Qwen3-shaped GGUF for browser-port verification (M1).

Weights are deterministic (numpy PRNG, fixed seed), so the native GPU path
and the browser path can be compared logit-for-logit on any machine without
shipping the real model.

    tools/.venv/bin/python tools/export_test_gguf.py --out testdata/tiny-qwen3-q8_0.gguf
"""
import argparse

import gguf
import numpy as np

# Qwen3 shape in miniature: every architectural feature present
# (GQA, QK-norm, SwiGLU, tied embeddings), every dimension tiny.
N_LAYERS = 2
HIDDEN = 64
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 16
FFN = 128
VOCAB = 512
THETA = 1000000.0
EPS = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="testdata/tiny-qwen3-q8_0.gguf")
    args = ap.parse_args()

    rng = np.random.default_rng(42)

    def t(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    w = gguf.GGUFWriter(args.out, "qwen3")
    w.add_block_count(N_LAYERS)
    w.add_context_length(256)
    w.add_embedding_length(HIDDEN)
    w.add_feed_forward_length(FFN)
    w.add_head_count(N_HEADS)
    w.add_head_count_kv(N_KV_HEADS)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_rope_freq_base(THETA)
    w.add_layer_norm_rms_eps(EPS)
    w.add_vocab_size(VOCAB)
    w.add_eos_token_id(0)

    from gguf import GGMLQuantizationType
    from gguf.quants import quantize

    def put(name, data):
        if data.ndim == 2:
            q = quantize(data, GGMLQuantizationType.Q8_0)
            w.add_tensor(name, q, raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, data)

    put("token_embd.weight", t(VOCAB, HIDDEN))
    for i in range(N_LAYERS):
        p = f"blk.{i}."
        put(p + "attn_norm.weight", np.ones(HIDDEN, np.float32) + t(HIDDEN, scale=0.01).reshape(-1))
        put(p + "attn_q.weight", t(N_HEADS * HEAD_DIM, HIDDEN))
        put(p + "attn_k.weight", t(N_KV_HEADS * HEAD_DIM, HIDDEN))
        put(p + "attn_v.weight", t(N_KV_HEADS * HEAD_DIM, HIDDEN))
        put(p + "attn_output.weight", t(HIDDEN, N_HEADS * HEAD_DIM))
        put(p + "attn_q_norm.weight", np.ones(HEAD_DIM, np.float32))
        put(p + "attn_k_norm.weight", np.ones(HEAD_DIM, np.float32))
        put(p + "ffn_norm.weight", np.ones(HIDDEN, np.float32))
        put(p + "ffn_gate.weight", t(FFN, HIDDEN))
        put(p + "ffn_up.weight", t(FFN, HIDDEN))
        put(p + "ffn_down.weight", t(HIDDEN, FFN))
    put("output_norm.weight", np.ones(HIDDEN, np.float32))
    # tied embeddings: no output.weight, same as the real export

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
