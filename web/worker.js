// nn browser worker — the production pipeline: WebGPU engine (m1.js) +
// wasm tokenizer (tokenizer.js) wired by chat.js. One GGUF feeds both.
// Verified end-to-end via Deno (web/_chat_test.deno.js): the browser stack
// reproduces the native qwen_chat reply.
//
// Serve the repo root so the relative fetches resolve, e.g.
//   python3 -m http.server 8080   →   http://localhost:8080/web/
// Override the GGUF location with ?model=<url> on the page URL (passed in
// via the "config" message) for a hosted demo (R2/HF) instead of the
// 640 MB file in parity/.

import { makeWasi } from "./wasi.js";
import { Chat } from "./chat.js";

const post = (m) => self.postMessage(m);
let chat = null;

// 640 MB Q8_0 GGUF (weights + tokenizer + sort permutation). Default is the
// same-origin local copy (works under `http.server` from the repo root);
// for a hosted demo pass ?model=<url> pointing at a CORS-enabled host
// (HuggingFace / R2 — GitHub release assets send NO CORS headers and can't
// be fetched cross-origin).
const DEFAULT_MODEL_URL = new URL("../parity/qwen3-0.6b-q8_0.gguf", import.meta.url).href;

async function boot(modelUrl) {
  try {
    if (!navigator.gpu) throw new Error("WebGPU がこのブラウザにありません");
    post({ type: "status", text: "GPUアダプタ取得中…" });
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("WebGPU adapter が取れません");
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 256 << 20),
        maxBufferSize: Math.min(lim.maxBufferSize, 256 << 20),
      },
    });
    device.lost.then((i) => post({ type: "error", text: "GPU device lost: " + i.message }));

    post({ type: "status", text: "カーネルとモデルを取得中…" });
    const [wgsl, tokWasm, gguf] = await Promise.all([
      fetch(new URL("../native/wgsl/qwen3.wgsl", import.meta.url)).then((r) => r.text()),
      fetch(new URL("./qwen_tokenizer.wasm", import.meta.url)).then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)),
      fetchProgress(modelUrl, (pct) => post({ type: "progress", pct })),
    ]);

    chat = await Chat.load({
      device, wgsl, gguf: gguf.buffer, tokWasm, makeWasi,
      onStatus: (s) => post({ type: "status", text: s }),
    });
    post({ type: "ready", text: "準備完了（100%ローカル）" });
  } catch (e) {
    post({ type: "error", text: String(e.message || e) });
  }
}

// fetch with a progress callback (the GGUF is large)
async function fetchProgress(url, onPct) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model fetch ${res.status} (${url})`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!total || !res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onPct((got / total) * 100);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "config") {
    boot(m.modelUrl || DEFAULT_MODEL_URL);
  } else if (m.type === "chat") {
    if (!chat) { post({ type: "error", text: "まだ読み込み中です" }); return; }
    try {
      await chat.generate(m.text, {
        temp: m.temp ?? 0.7, topP: m.topP ?? 0.9, maxNew: m.maxNew ?? 256,
        seed: (Math.random() * 2 ** 31) | 0,
        onToken: (chunk) => post({ type: "token", text: chunk }),
      });
      post({ type: "done" });
    } catch (err) {
      post({ type: "error", text: String(err.message || err) });
    }
  } else if (m.type === "reset") {
    chat?.reset();
    post({ type: "status", text: "文脈をクリアしました" });
  }
};
