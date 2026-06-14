// JS wrapper around the Almide Qwen3 tokenizer compiled to wasm
// (examples/qwen_tokenizer_wasm_entry.wasm). Browser/Deno/Node — the host
// just needs to supply a WASI import object (web/wasi.js makeWasi()).
//
// Almide heap-value ABI: 8-byte header [len:i32][cap:i32] then data, for
// Bytes / String / List[Int], params AND returns. Result = [tag:i32 @0]
// [payload_ptr:i32 @4], tag 0 = Ok. __alloc is a bump allocator (no free)
// → load the GGUF once, reuse the handle for every encode/decode.

export class Tokenizer {
  constructor(instance, handle) {
    this.ex = instance.exports;
    this.handle = handle;
  }

  static async load(wasmBytes, ggufBytes, makeWasi) {
    const wasi = makeWasi();
    const { instance } = await WebAssembly.instantiate(wasmBytes, wasi.imports);
    wasi.setMemory(instance.exports.memory);
    instance.exports._initialize?.();
    const t = new Tokenizer(instance, 0);
    const gptr = t._writeBytes(ggufBytes);
    t.handle = t._unwrap(t.ex.tok_load(gptr));
    return t;
  }

  _mem() {
    return this.ex.memory.buffer; // re-fetch: growth detaches the old buffer
  }
  _writeBytes(buf) {
    const ptr = this.ex.__alloc(8 + buf.length) >>> 0;
    const v = new DataView(this._mem());
    v.setInt32(ptr, buf.length, true);
    v.setInt32(ptr + 4, buf.length, true);
    new Uint8Array(this._mem()).set(buf, ptr + 8);
    return ptr;
  }
  _writeString(str) {
    return this._writeBytes(new TextEncoder().encode(str));
  }
  _writeListInt(arr) {
    const ptr = this.ex.__alloc(8 + arr.length * 8) >>> 0;
    const v = new DataView(this._mem());
    v.setInt32(ptr, arr.length, true);
    v.setInt32(ptr + 4, arr.length, true);
    for (let i = 0; i < arr.length; i++) v.setBigInt64(ptr + 8 + i * 8, BigInt(arr[i]), true);
    return ptr;
  }
  _readListInt(ptr) {
    const v = new DataView(this._mem());
    const p = ptr >>> 0;
    const len = v.getInt32(p, true);
    const out = new Array(len);
    for (let i = 0; i < len; i++) out[i] = Number(v.getBigInt64(p + 8 + i * 8, true));
    return out;
  }
  _readString(ptr) {
    const v = new DataView(this._mem());
    const p = ptr >>> 0;
    const len = v.getInt32(p, true);
    return new TextDecoder().decode(new Uint8Array(this._mem(), p + 8, len));
  }
  _unwrap(rawPtr) {
    const v = new DataView(this._mem());
    const p = rawPtr >>> 0;
    if (v.getInt32(p, true) !== 0) throw new Error("tokenizer: " + this._readString(v.getInt32(p + 4, true) >>> 0));
    return v.getInt32(p + 4, true) >>> 0;
  }

  encode(text) {
    return this._readListInt(this.ex.tok_encode(this.handle, this._writeString(text)));
  }
  decode(ids) {
    return this._readString(this.ex.tok_decode(this.handle, this._writeListInt(ids)));
  }
  // raw bytes of a single token (for UTF-8-safe streaming): decode([id]) as bytes
  special(str) {
    return Number(this.ex.tok_special(this.handle, this._writeString(str)));
  }
  get eos() {
    return Number(this.ex.tok_eos(this.handle));
  }
}
