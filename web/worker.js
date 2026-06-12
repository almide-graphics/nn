// nn browser worker — M0: device init + kernel compile + gemv smoke test
// via the shared module (web/m0.js).
//
// Serve from the REPO ROOT so the WGSL fetch resolves:
//   python3 -m http.server 8080      # then open http://localhost:8080/web/
//
// Milestones (docs/browser-port-plan.md):
//   M0 (now): prove the browser GPU path compiles our kernels and computes.
//   M1: tiny-model full layer stack from JS, logits vs native gpu_model.
//   M2: nn.wasm (Almide) owns gguf/tokenizer/op-list; this file degrades
//       to a dumb executor. Blocked on almide#643/#681.

import { initM0 } from "./m0.js";

const post = (m) => self.postMessage(m);

initM0((text) => post({ type: "status", text }))
  .then(() => post({ type: "ready", text: "M0 ok — kernels compile, gemv matches CPU" }))
  .catch((e) => post({ type: "error", text: String(e.message || e) }));

self.onmessage = (e) => {
  if (e.data.type === "chat") {
    // M2 wires this to nn.wasm. Until then, be honest about it.
    post({ type: "token", text: "（エンジン未接続 — M0スキャフォールドです。" });
    post({ type: "token", text: " カーネルのコンパイルとgemv検証は通っています）" });
    post({ type: "done" });
  }
};
