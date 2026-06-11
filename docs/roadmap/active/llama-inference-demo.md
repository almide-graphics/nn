<!-- description: End-to-end Qwen3 inference on Almide — L1 = logits parity vs HF fp32, then KV-cache + Q1_0 + WebGPU -->
# LLM Inference Demo (Qwen3)

Almide で decoder-only LLM を端から端まで動かし、「LLM が Almide 上で動く」を旗として示す arc。
Matrix perf arc (fusion stack + NumPy 勝ち) の投資回収段階で、Mission *"the language LLMs can write most accurately"* に直接効く成果物。

長期の出口は2つ:
1. **ローカルAIキャラ体験**: Whisper(済) → LLM(この arc) → nendo VRM → ceangal UI、全部ブラウザ
2. **Q1_0 (1-bit) × WebGPU**: bitnet.cpp が取っていない空白地帯

## 現在地 (2026-06-11 更新)

**ランタイム層はこの文書の旧 Stage 計画を追い越して実装済み。** `runtime/rs/src/matrix.rs`:

- `rms_norm_rows` / `swiglu_gate` / `silu_mul` — Llama 系プリミティブ一式
- `rope_rotate` / `rope_rotate_at(start_pos)` — **KVキャッシュ対応 API 形状で実装済み**
- `append_rows` — KVキャッシュ追記
- `qwen3_block_q1_0_kv` — Q1_0 量子化 + KVキャッシュ込みの Qwen3 ブロック融合呼び出し
- **Q1_0 = 自作 1-bit 量子化**(128要素ブロック、符号16バイト+scale、±scale)。
  `linear_q1_0_row_no_bias` は packed GGUF バイト列から直接 matmul(デコード割当なし)
- `per_head_rms_norm`(QK-norm) / `repeat_kv`(GQA) は **Rust 内部ヘルパーであり intrinsic 未公開**
  → nn 側は `split_cols_even` + `rms_norm_rows` + `concat_cols` で合成する(L1)。公開は perf 段階で検討

nn 側の資産:
- `gguf.almd` は汎用 GGUF パーサ(f32/f16)。ただし `extract_f32_matrix` は whisper 形式前提
  (dims[0]=rows)かつ要素単位読みで遅い。LLM ローダーは `from_bytes_f32_le/f16_le` +
  llama.cpp 規約(ne[0]=in, ne[1]=out → rows=dims[1], cols=dims[0])で別途実装する
- Llama 1-block: almide PR #218 (`spec/stdlib/matrix_llama_block_test.almd`, `examples/llama_block.almd`)

## ⚠️ コンパイラ制約と修正状況 (2026-06-12 深夜更新)

3つの独立したコンパイラ問題を特定。**全リリース版(〜v0.27.3)はこのパッケージを
ネイティブでビルドできない**。修正済みワークツリーのバイナリを使うこと:
`/tmp/almide-latest/target/release/almide` (branch `fix-bare-type-refs-in-lambdas`,
commit 92dd805b。codegen 107/107 + spec 265/265 通過)

1. **#433系: 裸の型名** (v0.26.16〜、修正済み): エイリアスimport型注釈 /
   `Option[自モジュール型]` 返却 / 自モジュールvariant / **ラムダparam・return型** /
   `Call.type_args` / `RcWrap.cast_ty` が正規化されず codegen に到達。
   修正: 入口での repair + リンク mangle の完備化 + 検証器の網羅。
   詳細: ワークツリーの `ISSUE_DRAFT_bare_type_names.md`
2. **records-of-Matrix がネイティブで不能** (多バージョン、修正済み):
   `AlmideRepr for AlmideMatrix` 欠落。`almide test` が WASM 優先なので発覚せず
3. **burn スプライスのマーカー腐敗** (flat移行で発生、ローカル回避のみ):
   `replace_matrix_runtime` の検出マーカーが flat-struct 移行後に kernel bridge の
   別の行へ誤マッチ → HEAD の全ネイティブ matrix ビルド破壊。本修正は flat 移行
   作業の領分(回避: スプライス skip)

パフォーマンス注意: 値セマンティクスのクローンが支配的(実測: 実行時間の84%が
memmove)。fold クロージャが Bytes を掴むと引数が by-value に降格して 2.4GB/call。
L2 の最優先 perf ターゲット(Matrix/Bytes の RcCow 化 or borrow 推定の強化)。

## 確定事項

- **被験体: Qwen3-0.6B**(hidden 1024 / 28 layers / 16 q-heads / 8 kv-heads / head_dim 128 /
  QK-norm / バイアスなし / tied embeddings / byte-level BPE / rope_theta 1e6 / rms_eps 1e-6)。
  ランタイム部品が Qwen3 の形をしている(QK-norm 等)。SentencePiece/protobuf 作業は不要になった
- **RoPE 規約**: `rope_rotate_at` は interleaved(GPT-J)方式 — (x[2i], x[2i+1]) 回転。
  HF Qwen3 は NeoX(rotate_half)方式。**ローダーで Q/K 重み行と q_norm/k_norm gamma を
  head 内 [j, j+half] → [2j, 2j+1] に並べ替えて吸収する**(q·k 内積はヘッド内同一置換に不変)
- **L1 合格基準: HF transformers fp32 と logits top-1 一致 ≥99% / 相対誤差 <1e-3**
  (固定トークンID列 20本、tokenizer はクリティカルパス外)。parity は f32 GGUF で取る
- Q1_0 融合パス(`qwen3_block_q1_0_kv`)は f32 で正解確立後の差別化レイヤー

## Stage 計画 (改訂)

### L1 — f32 正解パス (logits parity) ← いまここ

- [ ] `src/qwen.almd`: QwenLayer/QwenConfig 型、qwen_block(rms_norm → QKV → per-head QK-norm →
      rope_rotate_at → GQA expand → masked MHA → o_proj → 残差 → rms_norm → swiglu_gate → 残差)、
      qwen_forward、単体テスト
- [ ] `src/generate.almd`: `project()` を dot_row ループ → `linear_row_no_bias` 1 呼び出しに
      (Qwen vocab 151,936 で 15万 native call/step になるため。Whisper も共用で速くなる)
- [ ] `src/qwen_loader.almd`: GGUF → QwenModel。テンソル名マップ、メタデータ、RoPE 並べ替え
- [ ] `tools/dump_logits.py` + `examples/_parity_qwen3.almd` + 比較器
- [ ] greedy E2E (ID列→ID列、HF と 64 トークン照合はソフト基準)

### L2 — 速さの土台

- [ ] KVキャッシュ配線 (`rope_rotate_at` + `append_rows` は準備済み。attention 側 API 変更)
- [ ] サンプリング: top-k / top-p / temperature
- [ ] tokenizer: GGUF メタデータ(tokenizer.ggml.tokens/merges)から byte-BPE、ChatML テンプレート
- [ ] ストリーミングコールバック (almide-wasm-bindgen 経由)

### L3 — WebGPU

- [ ] WGSL GEMV/GEMM カーネル (snaidhm の tile dispatch 基盤と合流、`@gpu fn` 計画と整合)
- [ ] ブラウザで 0.6B が 30 tok/s 級

### L4 — Q1_0 差別化

- [ ] `qwen3_block_q1_0_kv` パスの精度評価 (BitNet b1.58 系チェックポイント)
- [ ] WebGPU 版 Q1_0 カーネル — bitnet.cpp(CPU専用)に対する空白地帯

### L5 — 公開

- [ ] README「LLM on Almide」、ブラウザデモ公開 URL、再現ベンチ
- [ ] ローカルAIキャラデモ統合 (Whisper + LLM + nendo + ceangal)

## 非ゴール

- 訓練 (forward のみ)
- Llama/Mistral 系の網羅対応 (Qwen3 1本。アーキ追加はデモ後)
- GGUF の全量子化型対応 (f32/f16 + Q1_0 のみ。Q8_0 は L2 で検討)

## 参照

- Whisper E2E: `docs/roadmap/active/whisper-almide.md`
- Matrix fusion stack: almide repo `docs/roadmap/active/mlir-backend-adoption.md`
