# llamacpp-chat2

> **開発中止:** このプロジェクトの開発は中止しました。新規の機能追加・修正は行いません。リポジトリはアーカイブ予定です。

ブラウザから使える LLM チャットです。GPU マシン上の Llama.cpp（`llama-server`）でモデルを動かし、画面から会話・モデルのダウンロード／読み込み／解放ができます。画像付きのやりとり（Vision）にも対応しています。

ローカルのほか、Paperspace / Google Colab 向け Notebook でも使えます。

---

## 構成

```text
[フロントエンド]                    [サーバーホスト]
  Gradio UI
  HttpBackend  ------chat/completions----->  llama-server (:8080)
  ControlClient -----制御API-------------->  control_api (:8090)
                                              |
                                              +-- process_manager
                                                    |
                                                    +-- llama-server 子プロセス
```

| 側 | 役割 |
|----|------|
| フロントエンド | Gradio 4.40 UI。推論・制御の HTTP クライアント。画像は multimodal で送信 |
| サーバー | 制御API（UI なし）。`llama-server` の load / unload / status / モデルダウンロード |

用語は **フロントエンド / サーバー** に統一しています（「バックエンド」とは呼びません）。

---

## 機能概要

- **Connection**
  - チャット用 URL（llama-server）とモデル操作 URL（制御API）、制御トークン
  - status / モデル一覧 / モデル追加（リポジトリ・LLM・任意 Vision・進捗表示）・一覧削除
  - Load / Unload / Restart（ngl・ctx、任意 mmproj）
- **Chat**
  - OpenAI 互換ストリーミング
  - 画像添付（ネイティブ Vision / mmproj 分離型の両方）。送信時に長い辺が上限（既定 1024px）を超える画像は縮小
  - メトリクス: 使用中/総 VRAM（GB）と返答時の tok/s
- **Vision**
  - ネイティブ統合 GGUF … mmproj **なし** で Load
  - mmproj 分離型 … 本体 + 任意 mmproj で Load

---

## 前提

- Python **3.10–3.13**（Gradio 4.40。3.14 は非対応）
- サーバー側に **CUDA 対応の `llama-server`**（[llama.cpp](https://github.com/ggml-org/llama.cpp) をビルド）
- GPU 推奨（Paperspace / Colab / ローカル）

---

## クイックスタート（ローカル）

リポジトリ直下で、**サーバー** と **フロントエンド** を別プロセスで起動します。

### 1. llama-server のパス

```bash
# Windows (cmd)
set LLAMACPP_CHAT2_LLAMA_SERVER=C:\path\to\llama-server.exe

# Linux / macOS
export LLAMACPP_CHAT2_LLAMA_SERVER=/path/to/llama-server
```

未設定の場合は `PATH` 上の `llama-server`、または `LLAMACPP_CHAT2_ROOT` 配下を探索します。

### 2. 制御API（サーバー）

```bat
setup_server.bat
```

```bash
./setup_server.sh
```

既定: `http://127.0.0.1:8090`、トークン `llamacpp-chat2`。追加 pip パッケージは不要です。

### 3. Gradio UI（フロントエンド）

別ターミナルで:

```bat
setup_frontend.bat
```

```bash
./setup_frontend.sh
```

既定: `http://127.0.0.1:7860`（ローカルでは `GRADIO_SHARE=False`）。

### 4. UI 操作

1. Connection で URL・トークンを確認（既定の 8080 / 8090 で同一マシン向け）
2. モデル追加（Hugging Face のリポジトリ URL + ファイル名など）→ 一覧更新 → Load
3. Chat で会話（画像は任意）

設定は `frontend-config.json` に保存できます。

---

## Paperspace / Google Colab

GPU ノートブック向け手順は次を使います。

| 環境 | Notebook |
|------|----------|
| Paperspace | [notebook/llamacpp-chat2-Paperspace.ipynb](notebook/llamacpp-chat2-Paperspace.ipynb) |
| Google Colab | [notebook/llamacpp-chat2-Colab.ipynb](notebook/llamacpp-chat2-Colab.ipynb) |

上から順に実行すると、`llama-server` のビルド、venv、制御API、Gradio（`share=True` / Public URL）まで進みます。

| ホスト | リポジトリ（永続） | モデル（一時） |
|--------|--------------------|----------------|
| Paperspace | `/notebooks/llamacpp-chat2/` | `/tmp/llamacpp-chat2/` |
| Colab | `/content/drive/MyDrive/llamacpp-chat2/` | `/content/llamacpp-chat2/` |

Colab は **ランタイム → GPU** を選択してください。ランタイム再起動後は venv 作成以降のセルを再実行します（Drive 上のリポジトリと llama.cpp ビルドは残ります）。

---

## 接続設定（2 URL）

| 項目 | 既定値 | 用途 |
|------|--------|------|
| チャット用 URL | `http://127.0.0.1:8080` | `llama-server`（会話） |
| モデル操作 URL | `http://127.0.0.1:8090` | 制御API |
| 制御APIトークン | `llamacpp-chat2` | ヘッダ `X-Control-Token`（または Bearer） |

リモート公開時はトンネル先 URL をそれぞれ設定します。トークンはフロントとサーバーで揃えてください。

主要な制御API:

| Method | Path |
|--------|------|
| GET | `/v1/control/status` |
| GET | `/v1/control/models` |
| POST | `/v1/control/models/download` |
| GET | `/v1/control/models/download/progress` |
| POST | `/v1/control/models/delete` |
| POST | `/v1/control/load` / `unload` / `restart` |

---

## 主な環境変数

| 変数 | 意味 |
|------|------|
| `LLAMACPP_CHAT2_LLAMA_SERVER` | `llama-server` 実行ファイル |
| `LLAMACPP_CHAT2_MODELS_DIR` | モデル一時ディレクトリ |
| `LLAMACPP_CHAT2_CONTROL_TOKEN` | 制御APIトークン |
| `LLAMACPP_CHAT2_CONTROL_HOST` / `_PORT` | 制御API バインド（既定 127.0.0.1:8090） |
| `LLAMACPP_CHAT2_LLAMA_HOST` / `_PORT` | 推論バインド（既定 127.0.0.1:8080） |
| `GRADIO_SHARE` | `True` で Public URL（Notebook 既定。ローカル setup は `False`） |
| `GRADIO_SERVER_NAME` / `_PORT` | Gradio バインド |
| `LLAMACPP_CHAT2_IMAGE_MAX_LONG_EDGE` | チャット送信時の画像長い辺上限 px（設定 JSON 未指定時。既定 1024） |

---

## ディレクトリ構成

```text
core/                 # Gradio 非依存（型・設定・HTTP クライアント・multimodal）
ui/                   # Gradio UI
server/               # 制御API・process_manager・モデルカタログ
notebook/             # Paperspace / Colab
setup_frontend.*      # フロント venv + UI 起動
setup_server.*        # サーバー venv + 制御API 起動
requirements-frontend.txt
requirements-server.txt   # 追加 pip なし（コメントのみ）
```

依存の境界:

| パッケージ | 依存してよいもの | 入れないもの |
|------------|------------------|--------------|
| `core/` | 標準ライブラリ、httpx 等 | Gradio、Forge |
| `ui/` | `core`、Gradio | `server/` 実装詳細 |
| `server/` | 標準ライブラリ、`llama-server` バイナリ | Gradio / FastAPI / torch / llama-cpp-python |

---

## テスト（サーバー）

```bash
python -m unittest server.test_server -v
```

---

## ライセンス

リポジトリにライセンスファイルがある場合はそれに従います。`llama.cpp` / 各モデルのライセンスは各配布元を確認してください。
