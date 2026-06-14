// M3 end-to-end: the full browser stack (wasm tokenizer + WebGPU engine)
// runs a real chat turn, greedy, and must match the native qwen_chat output
// token-for-token in spirit (same template, same model, same decode).
//
//   deno run --unstable-webgpu --allow-read web/_chat_test.deno.js
//
// Native reference (examples/qwen_chat.almd, greedy):
//   「日本の首都は？」 → 日本の首都は、**大阪市**です。

import { makeWasi } from "./wasi.js";
import { Chat } from "./chat.js";

const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) { console.log("SKIP no adapter"); Deno.exit(2); }
const lim = adapter.limits;
const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 256 << 20),
    maxBufferSize: Math.min(lim.maxBufferSize, 256 << 20),
  },
});

const wgsl = await Deno.readTextFile("native/wgsl/qwen3.wgsl");
const tokWasm = await Deno.readFile("examples/qwen_tokenizer_wasm_entry.wasm");
console.log("reading gguf…");
const gguf = (await Deno.readFile("parity/qwen3-0.6b-q8_0.gguf")).buffer;

const t0 = performance.now();
const chat = await Chat.load({ device, wgsl, gguf, tokWasm, makeWasi, onStatus: (s) => console.log("  " + s) });
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const prompt = "日本の首都は？";
console.log(`\n> ${prompt}`);
let streamed = "";
const t1 = performance.now();
const reply = await chat.generate(prompt, {
  temp: 0, // greedy → deterministic, matches native reference
  maxNew: 48,
  onToken: (c) => { streamed += c; Deno.stdout.writeSync(new TextEncoder().encode(c)); },
});
const dt = (performance.now() - t1) / 1000;
console.log(`\n\n(${(chat.pos / dt).toFixed(1)} tok/s incl. prefill)`);

const EXPECT_SUBSTR = "大阪市";
const ok = reply.includes(EXPECT_SUBSTR);
console.log(ok
  ? `M3_PASS — browser stack reproduced the native reply (contains "${EXPECT_SUBSTR}")`
  : `M3_FAIL — reply was: ${JSON.stringify(reply)}`);
Deno.exit(ok ? 0 : 1);
