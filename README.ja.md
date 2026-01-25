# ComfyUI-LLM-Session（日本語版）
[[en](README.md) | ja]

---

**Version:** 1.0.0  
**License:** GPL-3.0

**ComfyUI 上だけで動作するローカル LLM 実行環境**です。  
Ollama などの外部ランタイムを必要とせず、ComfyUI のノードとして完結します。

llama.cpp を通じて GGUF モデルをサポートしており、Qwen、Llama、Mistral NeMo、Gemma、Phi-3 Mini など、主要なオープン LLM を利用できます。

ユーザーとモデルの対話だけでなく、**モデル同士の対話**にも対応しており、
**観察・実験・分析**用途を主眼に設計されています。

---

## このプロジェクトでできること

- ローカル GGUF LLM 向けの **ファイルベース・セッション管理**
- **複数ターン会話を保持**できる ComfyUI ノード群
- モデルの挙動・収束・破綻を観察するための実験環境
- サーバーやデーモンを必要としない、ComfyUI 内完結構成

---

## このプロジェクトが目的としていないこと

- チャット UI ではありません
- API サーバーではありません
- Ollama の代替ではありません
- 一般ユーザー向けアプリケーションではありません

---

## 提供されるノード

### LLM Session Chat
セッションを保持する標準的なチャットノードです。すべての主要パラメータを調整できます。

### LLM Session Chat (Simple)
デモや初期検証向けの簡易版ノードです。

- セッション保持あり
- 内部で安全なデフォルト値を固定
- JSON 設定ファイルによる上書きが可能

### LLM Dialogue Cycle
モデル同士を対話させるためのノードです（グラフの循環は不要）。

### LLM Dialogue Cycle (Simple)
役割付き対話を観察することに特化した簡易版です。

---

## 設計の考え方

- **セッションファースト設計**：実行をまたいで会話が継続します
- **ファイルベース永続化**：外部 DB や状態管理は不要
- **観察重視**：対話ログ自体が主要な成果物です
- **決定論的なターン実行**：分析・比較に向いています

---

## インストール

### 1. リポジトリをクローン

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kantan-kanto/ComfyUI-LLM-Session.git
```

### 2. 依存関係のインストール

```bash
cd ComfyUI-LLM-Session
pip install -r requirements.txt
```

### 3. llama-cpp-python について

llama-cpp-python のビルド内容によって、対応モデルや Vision 機能の挙動が変わります。

- 公式ビルドは text-only 利用で安定
- Qwen3-VL などはカスタムビルドが必要な場合があります

詳細は **[COMPATIBILITY.md](COMPATIBILITY.md)** を参照してください。

### 4. モデル配置

GGUF モデルを以下に配置します。

```
ComfyUI/models/LLM/
```

---

## Simple ノードの重要設定（要点）

- **history_dir**：同じディレクトリを使う限り、会話が継続されます
- **config_path**：Simple ノードの内部デフォルトを JSON で上書きできます
- **force_text_only**（Dialogue Cycle Simple）：Vision 経路を無効化し、再現性を高めます

詳細設定は **[PARAMETERS.md](PARAMETERS.md)** を参照してください。

---

## サンプルワークフロー

すぐに試せるサンプルが含まれています。

```
examples/example_workflow.json
```

### 使い方

1. ComfyUI で [example_workflow.json](examples/example_workflow.json) を読み込む
2. GGUF モデルのパスを設定
3. `history_dir` を任意のディレクトリに設定
4. 1 回目を実行（Turn 1）
5. テキスト入力を Turn 2 のプロンプトに差し替えて再実行

2 回目の出力は、1 回目の実行内容に依存します。

---

## スクリーンショット

### LLM Session Chat (Simple)

![LLM Session Chat Simple](images/LLM_Session_Chat_Simple.png)

過去の会話履歴を保持しながら、ユーザーとモデルが対話している様子です。

### LLM Dialogue Cycle (Simple)

![LLM Dialogue Cycle Simple](images/LLM_Dialogue_Cycle_Simple.png)

モデル同士を対話させている様子です。
<br>ComfyUIでは、実行終了までアウトプットが出力されません。`history_dir`に保存されるテキストファイルをエディターなどで開いておくと、会話の進捗をリアルタイムに観察できます。

---

## ライセンス

本プロジェクトは **GNU General Public License v3.0** の下で公開されています。

llama-cpp-python への依存関係により、GPL-3.0 が適用されます。

---

## サポート

- 不具合報告・要望：GitHub Issues
- 詳細仕様：[README.md](README.md) / [PARAMETERS.md](PARAMETERS.md) / [COMPATIBILITY.md](COMPATIBILITY.md)
- サンプル：examples/ ディレクトリ

