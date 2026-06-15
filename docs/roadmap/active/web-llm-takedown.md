<!-- description: web-llm 打倒ロードマップ — FP16 Transformer正面を避け、①速度(三値/ゲート)②カテゴリ(喋るキャラ)③信頼性(監査可能)の3軸で包む。autograd→DLGN→ゲート数/深さPareto の研究成果を起点に -->
# Almide ロードマップ — web-llm 打倒 ＋ 検証可能ML回路共同設計

最終更新: 2026-06-15

> 旗: **web-llm を倒す。** ただし「FP16 Transformer を生tok/s で正面から」は
> 避ける（陳天奇/TVM の最強の一点）。彼らが弱い3方向から包む。

---

## 0. 現在地（証明済みの資産）

```mermaid
flowchart LR
    subgraph SHIPPED["出荷済み・検証済み"]
        E["LLM推論エンジン (nn)\nHF fp32 と logits 1e-6 一致\nllama.cpp 互角(CPU)"]
        B["ブラウザ完全ローカルチャット\nalmide-graphics.github.io/nn/web/\nエンジン+tokenizer 全部Almide"]
        AG["自作autograd\nscalar→tensor→DLGN, torch一致"]
        DL["DLGN 回路コスト最適化\nゲート数Pareto + 深さPareto\n多ターゲットbit等価(native/wasm/WGSL)"]
    end
    E --> B
    AG --> DL
```

---

## 1. web-llm を倒す3つの軸

```mermaid
flowchart TB
    GOAL(["web-llm を倒す"])

    subgraph A1["① 速度 — 正面・最難 (研究賭け)"]
        a1["FP16でなく軽量演算で勝つ\n三値(BitNet)/ゲート(DLGN)/matmul-free\n= 演算量 que 自体を変える\nTVMチューニングは三値に向いてない"]
    end
    subgraph A2["② カテゴリ — 横・最も確実 (実行)"]
        a2["web-llmはLLMランタイム単体\n喋るキャラ(音声+VRM+UI)全部Almide\n= 彼らが構造的に来れない場所"]
    end
    subgraph A3["③ 信頼性 — 縦・誰も主張してない (実行)"]
        a3["監査可能なローカルAI\nnative↔wasm↔WGSL bit等価\nlogits 1e-6 契約\n= 規制/エンタープライズ軸"]
    end

    GOAL --> A1
    GOAL --> A2
    GOAL --> A3

    NOTE["作戦: ①を狙う(当たれば決定打/外れても無傷)、\n②③で必ず勝つ。FP16正面だけは避ける"]
    A1 -.-> NOTE
    A2 -.-> NOTE
    A3 -.-> NOTE

    classDef bet fill:#3a341f,stroke:#aa3,color:#ffd
    classDef win fill:#1f3a24,stroke:#3a6,color:#dfd
    class A1 bet
    class A2,A3 win
```

---

## 2. 軸① 速度（研究線）— 今日のDLGN研究がここに繋がる

web-llmが速いのは「普通のTransformerだから」。ゲート/三値は演算が根本的に
軽い。それがブラウザで動けば、web-llmが最適化しても追いつけない領域がある。

```mermaid
flowchart LR
    R0["✅ DLGN 学習→硬化 (Almide autograd)"]
    R1["✅ 回路コスト(ゲート数)Pareto"]
    R2["✅ 多層scaling + 深さPareto"]
    R3["DLGN forward/backward を @gpu カーネルに\n(学習をGPUで, torchと同じ土俵)"]
    R4["三値(BitNet)をブラウザWebGPUで\n= 世界初, bitnet.cppはCPU専用\nQ8エンジン既にある→次の量子化ステップ"]
    R5["geata: Almide→ゲートnetlist/HDL出力\n= 硬化回路を実シリコンへ (Strike2)"]
    R0 --> R1 --> R2 --> R3
    R2 --> R4
    R2 --> R5
    style R0 fill:#1f3a24,stroke:#3a6,color:#dfd
    style R1 fill:#1f3a24,stroke:#3a6,color:#dfd
    style R2 fill:#1f3a24,stroke:#3a6,color:#dfd
```

**速度の留保（正直に）**: 学習探索は torch が速い → torch をオラクルに使い続ける。
Almideの速度的価値は「学習」でなく「**検証可能な多ターゲット推論/デプロイ**」。
torch と同じトラックで競争しない。

---

## 3. 軸② カテゴリ（実行線）— 喋るローカルAIキャラ

部品は全部ある。統合するだけ。web-llm が原理的に作れないもの。

```mermaid
flowchart LR
    V0["✅ ブラウザLLM (nn/web)"]
    V1["音声入力: Web Speech API (SpeechRecognition)"]
    V2["音声出力: SpeechSynthesis (日本語)"]
    V3["VRMアバター: nendo"]
    V4["UI: ceangal"]
    V5["★ 喋るローカルAIキャラ\n= マイクで話す→AIが考える→VRMが声で返す\n全部ブラウザ・全部Almide・全部ローカル"]
    V0 --> V5
    V1 --> V5
    V2 --> V5
    V3 --> V5
    V4 --> V5
    style V0 fill:#1f3a24,stroke:#3a6,color:#dfd
    style V5 fill:#1f2a3a,stroke:#39c,color:#def
```

注意: VRM(nendo)とWhisperのブラウザ移植は未検証。音声I/O(Web Speech)は
ブラウザ標準で即実現可能 → **まず音声I/Oから**、VRMは次フェーズ。

---

## 4. 軸③ 信頼性（実行線）— 監査可能なローカルAI

```mermaid
flowchart LR
    T0["✅ native↔wasm↔WGSL bit等価 (DLGN)"]
    T1["✅ logits 1e-6 契約 (LLMエンジン)"]
    T2["契約台帳を第一級の証明物に\n= クロスターゲット等価性を売りにする"]
    T3["MSR (modification survival rate)\n= LLMネイティブ言語の領域 (dojo計測)"]
    T0 --> T2
    T1 --> T2
    T2 --> T3
    style T0 fill:#1f3a24,stroke:#3a6,color:#dfd
    style T1 fill:#1f3a24,stroke:#3a6,color:#dfd
```

---

## 5. 直近の優先順（次やること）

```mermaid
flowchart TB
    P1["A. 音声I/O 統合 (軸②, 数日, 確実, SNS映え最大)\n話しかける→ローカルAIが声で返す"]
    P2["B. DLGN @gpu カーネル化 (軸①, 速度の前提)\nまずforward/backwardをmatrix/WGSLへ"]
    P3["C. 記事化 (信用作り)\nautograd→DLGN→ゲート数/深さPareto, 再現手順つき\nキラーフレーズ: 自作言語で自動微分から論理回路共同設計まで, PyTorch一致で検証"]
    P4["D. 三値ブラウザ実証 (軸①本命, 研究賭け)\nBitNet GGUF→三値WGSLカーネル→ブラウザ"]
    P5["E. geata: Almide→netlist (Strike2インフラ)"]

    P1 --> P3
    P2 --> P4
    P3 -.信用.-> P4
    P4 --> P5

    style P1 fill:#1f2a3a,stroke:#39c,color:#def
    style P3 fill:#1f2a3a,stroke:#39c,color:#def
```

推奨初手: **A(音声I/O) と C(記事化) を並行**。Aは確実に映える・部品あり、
Cは今日までの研究を信用に変える。その後 B→D で速度の本命(軸①)へ。

---

## 6. 言ってはいけないこと（誇張封じ）

- 「CPU生tok/sで llama.cpp を超えた」→ NG。正: 「同条件で互角＋熱耐性で上」
- 「web-llmより速い」→ まだNG (軸①が当たるまで)。今言えるのは軸②③
- 「既存LLMより賢い」→ NG。これは効率・検証可能性・回路最適化の話
- モデルが小さい(0.6/1.7B)ので賢さは限定的、と先に線を引く

> 強い人は事実で積む。1桁以内の事実で勝負する方が、専門家には最強。
