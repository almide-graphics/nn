// In-browser Qwen3 chat: ties the wasm tokenizer (web/tokenizer.js) to the
// WebGPU engine (web/m1.js). One GGUF feeds both — m1.js reads the weights,
// the tokenizer wasm reads the tokenizer.ggml.* section.
//
// Mirrors examples/qwen_chat.almd: Qwen3 chat template with an empty
// <think> block (non-thinking), positional KV cache across turns, greedy
// or temperature/top-p sampling, UTF-8-safe streaming.

import { Tokenizer } from "./tokenizer.js";
import { loadModel } from "./m1.js";

export class Chat {
  constructor(tok, model) {
    this.tok = tok;
    this.model = model;
    this.pos = 0;
    this.maxSeq = 2048;
    // special-token ids for the template
    this.imStart = tok.special("<|im_start|>");
    this.imEnd = tok.special("<|im_end|>");
    this.thinkOpen = tok.special("<think>");
    this.thinkClose = tok.special("</think>");
    this.eos = tok.eos;
  }

  static async load({ device, wgsl, gguf, tokWasm, makeWasi, onStatus = () => {} }) {
    onStatus("loading tokenizer…");
    const tok = await Tokenizer.load(tokWasm, new Uint8Array(gguf), makeWasi);
    onStatus("uploading weights to GPU…");
    const model = await loadModel(device, wgsl, gguf, onStatus);
    return new Chat(tok, model);
  }

  reset() {
    this.pos = 0;
  }

  // Token ids for one user turn (template). When pos>0 we first close the
  // previous assistant message, exactly like the Almide REPL.
  _turnIds(userText) {
    const t = this.tok;
    const ids = [];
    if (this.pos > 0) {
      ids.push(this.imEnd, ...t.encode("\n"));
    }
    ids.push(this.imStart, ...t.encode("user\n" + userText), this.imEnd);
    ids.push(...t.encode("\n"), this.imStart, ...t.encode("assistant\n"));
    if (this.thinkOpen >= 0 && this.thinkClose >= 0) {
      ids.push(this.thinkOpen, ...t.encode("\n\n"), this.thinkClose, ...t.encode("\n\n"));
    }
    return ids;
  }

  // Generate a reply, streaming decoded text to onToken(textChunk).
  // sampling: {temp, topP, seed} — temp<=0 = greedy. Returns the full text.
  async generate(userText, { maxNew = 256, temp = 0.7, topP = 0.9, seed = 1, onToken = () => {} } = {}) {
    const t = this.tok;
    let ids = this._turnIds(userText);
    if (this.pos > 0 && this.pos + ids.length + maxNew + 16 > this.maxSeq) {
      this.reset();
      ids = this._turnIds(userText);
    }

    // prefill; the last step's logits give the first generated token
    let logits = null;
    for (const tk of ids) {
      logits = await this.model.step(tk, this.pos);
      this.pos++;
    }
    let next = pick(logits, temp, topP, seed, this.pos);

    const pending = [];
    let full = "";
    let rng = seed;
    for (let k = 0; k < maxNew; k++) {
      if (next === this.imEnd || next === this.eos || next < 0) break;
      // raw bytes of this token (no per-token string decode — that corrupts
      // a multibyte char split across tokens), stream UTF-8-safe
      for (const b of t.tokenBytes(next)) pending.push(b);
      const cut = utf8CompleteLen(pending);
      if (cut > 0) {
        const chunk = new TextDecoder().decode(new Uint8Array(pending.splice(0, cut)));
        full += chunk;
        onToken(chunk);
      }
      logits = await this.model.step(next, this.pos);
      this.pos++;
      rng = (rng + 0x9e3779b9) | 0;
      next = pick(logits, temp, topP, rng, this.pos);
    }
    if (pending.length) {
      const chunk = new TextDecoder().decode(new Uint8Array(pending));
      full += chunk;
      onToken(chunk);
    }
    return full;
  }
}

// Longest prefix of bs that is complete UTF-8 (hold back a split tail).
function utf8CompleteLen(bs) {
  const n = bs.length;
  let cut = n;
  let scanned = 0;
  for (let i = n - 1; i >= 0 && scanned < 4; i--, scanned++) {
    const b = bs[i];
    if (b < 0x80) break;
    if (b >= 0xc0) {
      const need = b >= 0xf0 ? 4 : b >= 0xe0 ? 3 : 2;
      if (n - i < need) cut = i;
      break;
    }
  }
  return cut;
}

function argmax(a) {
  let b = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
  return b;
}

// temperature + top-p sampling (greedy when temp<=0). Deterministic per seed.
function pick(logits, temp, topP, seed, pos) {
  if (temp <= 0) return argmax(logits);
  const K = 256;
  const idx = Array.from(logits.keys());
  idx.sort((a, b) => logits[b] - logits[a]);
  const top = idx.slice(0, K);
  const m = logits[top[0]];
  const inv = 1 / temp;
  const ps = top.map((i) => Math.exp((logits[i] - m) * inv));
  const sum = ps.reduce((a, b) => a + b, 0);
  const cut = Math.min(1, Math.max(0, topP)) * sum;
  let nKeep = ps.length;
  let acc = 0;
  for (let i = 0; i < ps.length; i++) {
    acc += ps[i];
    if (acc >= cut) { nKeep = i + 1; break; }
  }
  // xorshift32 from (seed, pos)
  let s = (seed ^ (pos * 0x85ebca6b)) | 0;
  s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
  const r = ((s >>> 0) / 0x100000000) * ps.slice(0, nKeep).reduce((a, b) => a + b, 0);
  let acc2 = 0;
  for (let i = 0; i < nKeep; i++) {
    acc2 += ps[i];
    if (r <= acc2) return top[i];
  }
  return top[nKeep - 1];
}
