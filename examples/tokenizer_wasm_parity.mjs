#!/usr/bin/env node
// M2 acceptance: tokenize the 20 parity prompts INSIDE the wasm module
// (Almide tokenizer compiled to wasm, called from JS the way the browser
// will) and diff against the HF ground truth in parity/token_ids_N.txt.
//
//   node examples/tokenizer_wasm_parity.mjs
//
// Proves the full Almide tokenizer — GGUF parse, 152k sorted dicts, BPE
// merges, byte-level encode — runs correctly on the wasm target with the
// JS-hands-the-bytes-in ABI that the browser uses.

import { WASI } from 'node:wasi';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, '..');

// Same 20 prompts as tools/dump_logits.py (source of token_ids_N.txt).
const PROMPTS = [
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
];

const wasmBuf = await readFile(resolve(__dirname, 'qwen_tokenizer_wasm_entry.wasm'));
const gguf = await readFile(resolve(root, 'parity/qwen3-0.6b-q8_0.gguf'));
console.log(`[node] wasm ${wasmBuf.length} B, gguf ${(gguf.length / 1048576).toFixed(0)} MB`);

const wasi = new WASI({ version: 'preview1', args: [], env: {}, preopens: {} });
const { instance } = await WebAssembly.instantiate(wasmBuf, wasi.getImportObject());
wasi.initialize(instance);

const ex = instance.exports;
const alloc = ex.__alloc;
let mem = () => ex.memory.buffer; // re-fetch after every alloc (growth detaches)

// Almide heap-value ABI: 8-byte header [len:i32][cap:i32] then data
// (empirically confirmed for Bytes/String/List params; returns share it).
function writeBytes(buf) {
  const ptr = alloc(8 + buf.length) >>> 0;
  const v = new DataView(mem());
  v.setInt32(ptr, buf.length, true);
  v.setInt32(ptr + 4, buf.length, true);
  new Uint8Array(mem()).set(buf, ptr + 8);
  return ptr;
}
function writeString(str) {
  return writeBytes(new TextEncoder().encode(str));
}
function writeListInt(arr) {
  const ptr = alloc(8 + arr.length * 8) >>> 0;
  const v = new DataView(mem());
  v.setInt32(ptr, arr.length, true);
  v.setInt32(ptr + 4, arr.length, true);
  for (let i = 0; i < arr.length; i++) v.setBigInt64(ptr + 8 + i * 8, BigInt(arr[i]), true);
  return ptr;
}
function readListInt(ptr) {
  const v = new DataView(mem());
  const p = ptr >>> 0;
  const len = v.getInt32(p, true);
  const out = new Array(len);
  for (let i = 0; i < len; i++) out[i] = Number(v.getBigInt64(p + 8 + i * 8, true));
  return out;
}
function readString(ptr) {
  const v = new DataView(mem());
  const p = ptr >>> 0;
  const len = v.getInt32(p, true);
  return new TextDecoder().decode(new Uint8Array(mem(), p + 8, len));
}
// Result[T,String] = [tag:i32][payload:i32]; tag 0 = Ok
function unwrap(rawPtr) {
  const v = new DataView(mem());
  const ptr = rawPtr >>> 0;
  const tag = v.getInt32(ptr, true);
  const payload = v.getInt32(ptr + 4, true) >>> 0;
  if (tag !== 0) throw new Error('Almide Err: ' + readString(payload));
  return payload;
}

// 1. load the tokenizer ONCE → handle
let t0 = Date.now();
const ggufPtr = writeBytes(gguf);
const handle = unwrap(ex.tok_load(ggufPtr));
console.log(`[node] tok_load: ${((Date.now() - t0) / 1000).toFixed(1)}s, handle=${handle}`);

// 2. encode every prompt against the one handle, diff vs HF
let pass = 0, fail = 0;
t0 = Date.now();
for (let i = 0; i < PROMPTS.length; i++) {
  const textPtr = writeString(PROMPTS[i]);
  const ids = readListInt(ex.tok_encode(handle, textPtr));
  const want = (await readFile(resolve(root, `parity/token_ids_${i}.txt`), 'utf8')).trim();
  const got = ids.join(',');
  const idsMatch = got === want;

  // round-trip decode
  const idsPtr = writeListInt(ids);
  const back = readString(ex.tok_decode(handle, idsPtr));
  const rtMatch = back === PROMPTS[i];

  if (idsMatch && rtMatch) {
    pass++;
    console.log(`[${i}] OK (${ids.length} tokens)`);
  } else {
    fail++;
    console.log(`[${i}] MISMATCH${idsMatch ? '' : ' ids'}${rtMatch ? '' : ' roundtrip'}`);
    if (!idsMatch) { console.log(`   want: ${want}`); console.log(`   got : ${got}`); }
    if (!rtMatch) console.log(`   decode: ${JSON.stringify(back)}`);
  }
}
console.log(`[node] encode 20 prompts: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
console.log(`\n${pass}/${pass + fail} prompts match HF tokenization IN WASM`);
process.exit(fail === 0 ? 0 : 1);
