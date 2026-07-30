"""Gradio 4.40 フロントエンドのエントリポイント。"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from core.config import default_config_path, load_config
from ui import components as C


def _repo_root() -> Path:
    """リポジトリルートを返す。

    Returns
    -------
    pathlib.Path
        ``ui/`` の親ディレクトリ。
    """
    return Path(__file__).resolve().parent.parent


def _env_flag(name: str, *, default: bool = False) -> bool:
    """環境変数を真偽値として読む。

    Parameters
    ----------
    name : str
        変数名。
    default : bool, default False
        未設定時の値。

    Returns
    -------
    bool
        ``1`` / ``true`` / ``yes`` / ``on`` なら True。
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def build_ui(config_path: Path | None = None) -> gr.Blocks:
    """メイン UI を構築する。

    Parameters
    ----------
    config_path : pathlib.Path or None, default None
        設定 JSON のパス。省略時はリポジトリ直下の既定ファイル。

    Returns
    -------
    gradio.Blocks
        組み立て済み Blocks。
    """
    cfg_path = config_path or default_config_path(_repo_root())
    cfg = load_config(cfg_path)

    with gr.Blocks(title="llamacpp-chat2") as demo:
        gr.Markdown("# llamacpp-chat2")
        config_path_state = gr.State(str(cfg_path))

        with gr.Tabs():
            with gr.Tab("Connection"):
                with gr.Row():
                    inference_url = gr.Textbox(
                        label="チャット用 URL（llama-server）",
                        info="会話の送信先。同一マシン例: http://127.0.0.1:8080",
                        value=cfg.inference_base_url,
                        scale=2,
                        placeholder="http://127.0.0.1:8080",
                    )
                    control_url = gr.Textbox(
                        label="モデル操作 URL（制御API）",
                        info="ロード／アンロード／status。同一マシン例: http://127.0.0.1:8090",
                        value=cfg.control_base_url,
                        scale=2,
                        placeholder="http://127.0.0.1:8090",
                    )
                token = gr.Textbox(
                    label="制御APIトークン",
                    info=(
                        "サーバーの LLAMACPP_CHAT2_CONTROL_TOKEN と同じ値。"
                        "既定: llamacpp-chat2"
                    ),
                    value=cfg.control_token,
                    type="password",
                    placeholder="llamacpp-chat2",
                )
                with gr.Row():
                    ngl = gr.Number(
                        label="ngl (--n-gpu-layers)",
                        value=cfg.ngl,
                        precision=0,
                    )
                    ctx = gr.Number(
                        label="ctx (--ctx-size)",
                        value=cfg.ctx,
                        precision=0,
                    )
                    timeout_sec = gr.Number(
                        label="タイムアウト（秒）",
                        value=cfg.timeout_sec,
                        precision=0,
                    )
                with gr.Row():
                    btn_save = gr.Button("設定を保存")
                    btn_status = gr.Button("Status / health")
                    btn_refresh = gr.Button("モデル一覧を更新")
                log = gr.Textbox(label="Connection ログ", lines=12)

                gr.Markdown("### モデル追加（ダウンロード）")
                with gr.Row():
                    model_id_in = gr.Textbox(
                        label="タイトル（ID）",
                        placeholder="Foo Q4_K",
                        scale=1,
                    )
                    model_url_in = gr.Textbox(
                        label="リポジトリ",
                        placeholder="org/repo または https://huggingface.co/...",
                        scale=2,
                    )
                with gr.Row():
                    model_filename_in = gr.Textbox(
                        label="LLMモデル",
                        placeholder="Foo-Q4_K.gguf",
                        scale=1,
                    )
                    model_mmproj_in = gr.Textbox(
                        label="Visionモデル",
                        placeholder="mmproj-model-bf16.gguf（任意・空可）",
                        scale=1,
                    )
                with gr.Row():
                    model_overwrite = gr.Checkbox(
                        label="上書きする", value=False
                    )
                    btn_download = gr.Button(
                        "ダウンロード", variant="secondary"
                    )

                with gr.Row():
                    model_dd = gr.Dropdown(
                        label="モデル（GGUF）",
                        choices=[],
                        value=None,
                        allow_custom_value=True,
                        scale=3,
                    )
                    btn_delete_model = gr.Button(
                        "リストから削除", variant="stop", scale=1
                    )
                with gr.Row():
                    mmproj_dd = gr.Dropdown(
                        label="mmproj（任意・空＝ネイティブ Vision / テキスト専用）",
                        choices=[""],
                        value="",
                        allow_custom_value=True,
                        scale=3,
                    )
                    btn_delete_mmproj = gr.Button(
                        "Visionを削除", variant="stop", scale=1
                    )
                with gr.Row():
                    btn_load = gr.Button("Load", variant="primary")
                    btn_unload = gr.Button("Unload / Stop")
                    btn_restart = gr.Button("Restart")
                    btn_start = gr.Button("Start")

                conn_inputs = [
                    inference_url,
                    control_url,
                    token,
                    ngl,
                    ctx,
                    timeout_sec,
                ]

                btn_save.click(
                    fn=lambda *a: C.save_connection(*a[:-1], a[-1]),
                    inputs=conn_inputs + [config_path_state],
                    outputs=log,
                )
                btn_status.click(
                    fn=C.do_status, inputs=conn_inputs, outputs=log
                )
                btn_refresh.click(
                    fn=C.refresh_models,
                    inputs=conn_inputs,
                    outputs=[model_dd, mmproj_dd, log],
                )
                btn_download.click(
                    fn=C.download_and_refresh,
                    inputs=conn_inputs
                    + [
                        model_id_in,
                        model_url_in,
                        model_filename_in,
                        model_mmproj_in,
                        model_overwrite,
                    ],
                    outputs=[model_dd, mmproj_dd, log],
                )
                btn_delete_model.click(
                    fn=C.delete_and_refresh,
                    inputs=conn_inputs + [model_dd],
                    outputs=[model_dd, mmproj_dd, log],
                )
                btn_delete_mmproj.click(
                    fn=C.delete_and_refresh,
                    inputs=conn_inputs + [mmproj_dd],
                    outputs=[model_dd, mmproj_dd, log],
                )
                btn_load.click(
                    fn=lambda *a: C.ctrl_action(
                        "load",
                        *a[:-2],
                        model=a[-2] or "",
                        mmproj=a[-1] or "",
                    ),
                    inputs=conn_inputs + [model_dd, mmproj_dd],
                    outputs=log,
                )
                btn_unload.click(
                    fn=lambda *a: C.ctrl_action("unload", *a),
                    inputs=conn_inputs,
                    outputs=log,
                )
                btn_restart.click(
                    fn=lambda *a: C.ctrl_action("restart", *a),
                    inputs=conn_inputs,
                    outputs=log,
                )
                btn_start.click(
                    fn=lambda *a: C.ctrl_action(
                        "start",
                        *a[:-2],
                        model=a[-2] or None,
                        mmproj=a[-1] or "",
                    ),
                    inputs=conn_inputs + [model_dd, mmproj_dd],
                    outputs=log,
                )

            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(label="チャット", height=480)
                metrics = gr.Textbox(
                    label="メトリクス",
                    value="VRAM — · — tok/s",
                    interactive=False,
                    lines=1,
                )
                msg = gr.MultimodalTextbox(
                    label="メッセージ（画像は Vision 対応モデル向け）",
                    file_types=["image"],
                    file_count="multiple",
                    lines=3,
                )
                temperature = gr.Slider(
                    0.0, 2.0, value=0.7, step=0.05, label="temperature"
                )
                image_max_long_edge = gr.Number(
                    label="画像 長辺上限 (px)",
                    info="長い辺がこの値を超える画像は送信時に縮小（0 以下で無制限）",
                    value=cfg.image_max_long_edge,
                    precision=0,
                )
                with gr.Row():
                    btn_send = gr.Button("送信", variant="primary")
                    btn_clear = gr.Button("クリア")

                def send_wrap(
                    message,
                    history,
                    inf,
                    ctl,
                    tok,
                    ngl_v,
                    ctx_v,
                    to,
                    temp,
                    max_edge,
                ):
                    """チャット送信ラッパ。"""
                    yield from C.chat_respond(
                        message,
                        history,
                        inf,
                        ctl,
                        tok,
                        ngl_v,
                        ctx_v,
                        to,
                        temp,
                        image_max_long_edge=max_edge,
                    )

                btn_send.click(
                    fn=send_wrap,
                    inputs=[msg, chatbot]
                    + conn_inputs
                    + [temperature, image_max_long_edge],
                    outputs=[chatbot, metrics],
                ).then(C.empty_multimodal, None, msg)
                msg.submit(
                    fn=send_wrap,
                    inputs=[msg, chatbot]
                    + conn_inputs
                    + [temperature, image_max_long_edge],
                    outputs=[chatbot, metrics],
                ).then(C.empty_multimodal, None, msg)
                btn_clear.click(fn=lambda: [], outputs=chatbot)

        gr.Markdown(
            f"設定ファイル: `{cfg_path}` · Gradio 4.40 · "
            "固まった場合は Connection の unload / restart / load で回復。"
        )

    return demo


def main() -> None:
    """UI を起動する。"""
    os.environ.setdefault("PYTHONUTF8", "1")
    root = _repo_root()
    demo = build_ui(default_config_path(root))
    demo.queue()
    share = _env_flag("GRADIO_SHARE", default=True)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=share,
    )


if __name__ == "__main__":
    main()
