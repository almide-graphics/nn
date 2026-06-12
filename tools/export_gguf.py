#!/usr/bin/env python3
"""Export HF Qwen3 weights to an f32 GGUF with llama.cpp tensor naming.

Produces the exact subset of tensors/metadata that nn.qwen_loader reads.
Writing our own f32 file (instead of downloading a community F16 GGUF)
removes quantization noise from the parity comparison entirely.

Usage:
    tools/.venv/bin/python tools/export_gguf.py --out parity/qwen3-0.6b-f32.gguf
"""
import argparse

import gguf
import torch
from transformers import AutoConfig, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="parity/qwen3-0.6b-f32.gguf")
    ap.add_argument("--quant", choices=["f32", "q8_0"], default="f32",
                    help="q8_0 quantizes 2-D weight tensors (norm gammas stay f32)")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    sd = model.state_dict()

    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    w = gguf.GGUFWriter(args.out, "qwen3")
    w.add_block_count(cfg.num_hidden_layers)
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size)
    w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(cfg.num_attention_heads)
    w.add_head_count_kv(cfg.num_key_value_heads)
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_vocab_size(cfg.vocab_size)
    # eos/bos as plain metadata so the Almide side can read them without
    # a full tokenizer section
    w.add_eos_token_id(getattr(cfg, "eos_token_id", 151645) or 151645)

    def put(name: str, key: str) -> None:
        data = sd[key].float().numpy()
        if args.quant == "q8_0" and data.ndim == 2:
            from gguf import GGMLQuantizationType
            from gguf.quants import quantize
            q = quantize(data, GGMLQuantizationType.Q8_0)
            # No raw_shape: gguf derives the logical shape from the quantized
            # byte shape (last dim / 34 * 32).
            w.add_tensor(name, q, raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, data)

    put("token_embd.weight", "model.embed_tokens.weight")
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        put(f"blk.{i}.attn_norm.weight", p + "input_layernorm.weight")
        put(f"blk.{i}.attn_q.weight", p + "self_attn.q_proj.weight")
        put(f"blk.{i}.attn_k.weight", p + "self_attn.k_proj.weight")
        put(f"blk.{i}.attn_v.weight", p + "self_attn.v_proj.weight")
        put(f"blk.{i}.attn_output.weight", p + "self_attn.o_proj.weight")
        put(f"blk.{i}.attn_q_norm.weight", p + "self_attn.q_norm.weight")
        put(f"blk.{i}.attn_k_norm.weight", p + "self_attn.k_norm.weight")
        put(f"blk.{i}.ffn_norm.weight", p + "post_attention_layernorm.weight")
        put(f"blk.{i}.ffn_gate.weight", p + "mlp.gate_proj.weight")
        put(f"blk.{i}.ffn_up.weight", p + "mlp.up_proj.weight")
        put(f"blk.{i}.ffn_down.weight", p + "mlp.down_proj.weight")
    put("output_norm.weight", "model.norm.weight")
    # tied embeddings: no output.weight on purpose

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
