"""Gradio UI 用の Connection / Chat ハンドラ。"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Generator, Optional, Union

import gradio as gr

from core.backend.control_client import ControlClientError
from core.backend.factory import make_control_client, make_http_backend
from core.backend.http import HttpBackendError
from core.config import save_config
from core.errors import format_error
from core.multimodal import (
    build_user_content,
    chatbot_user_display,
    extract_user_turn,
    is_mmproj_name,
)
from core.types import ChatMessage, ConnectionConfig

# MultimodalTextbox: dict、従来 Textbox: str
ChatInput = Union[str, dict[str, Any], None]


def cfg_from_inputs(
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
) -> ConnectionConfig:
    """UI 入力から接続設定を組み立てる。

    Parameters
    ----------
    inference_url : str
        チャット用 URL（llama-server）。
    control_url : str
        モデル操作 URL（制御API）。
    token : str
        制御APIトークン。
    ngl : float or int
        GPU 層数。
    ctx : float or int
        コンテキスト長。
    timeout_sec : float
        タイムアウト秒。

    Returns
    -------
    ConnectionConfig
        設定オブジェクト。
    """
    return ConnectionConfig(
        inference_base_url=(inference_url or "").strip()
        or "http://127.0.0.1:8080",
        control_base_url=(control_url or "").strip()
        or "http://127.0.0.1:8090",
        control_token=(token or "").strip()
        or ConnectionConfig.control_token,
        ngl=int(ngl),
        ctx=int(ctx),
        timeout_sec=float(timeout_sec),
        stream_timeout_sec=max(float(timeout_sec) * 5.0, 300.0),
    )


def save_connection(
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
    config_path: str,
) -> str:
    """接続設定をファイルに保存する。

    Returns
    -------
    str
        保存結果メッセージ。
    """
    cfg = cfg_from_inputs(
        inference_url, control_url, token, ngl, ctx, timeout_sec
    )
    path = save_config(cfg, config_path)
    return f"保存しました: {path}"


def do_status(
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
) -> str:
    """制御 status と推論 health をまとめて表示用文字列にする。

    Returns
    -------
    str
        ログテキスト。
    """
    cfg = cfg_from_inputs(
        inference_url, control_url, token, ngl, ctx, timeout_sec
    )
    lines: list[str] = []
    ctrl = make_control_client(cfg)
    try:
        st = ctrl.status()
        lines.append(
            "制御 status: ok={ok} llama_running={run} model={model}".format(
                ok=st.ok,
                run=st.llama_running,
                model=st.loaded_model or "(なし)",
            )
        )
        if (
            st.vram_used_gb is not None
            and st.vram_total_gb is not None
        ):
            lines.append(
                f"VRAM: {st.vram_used_gb:.1f} / {st.vram_total_gb:.1f} GB"
            )
        if st.message:
            lines.append(f"message: {st.message}")
        if st.exit_code is not None:
            lines.append(f"exit_code: {st.exit_code}")
        if st.log_tail:
            lines.append("--- log_tail ---\n" + st.log_tail[-2000:])
    except ControlClientError as e:
        lines.append(f"制御エラー:\n{e}")

    http = make_http_backend(cfg)
    try:
        healthy = http.health()
        lines.append(f"推論 health: {'ok' if healthy else 'failed'}")
    except HttpBackendError as e:
        lines.append(f"推論エラー:\n{e}")

    return "\n".join(lines)


def _split_model_labels(labels: list[str]) -> tuple[list[str], list[str]]:
    """モデル一覧を本体 GGUF と mmproj に分ける。

    Parameters
    ----------
    labels : list of str
        サーバーから得たラベル。

    Returns
    -------
    models : list of str
        チャット用モデル（mmproj 以外）。
    mmprojs : list of str
        プロジェクター候補。
    """
    models: list[str] = []
    mmprojs: list[str] = []
    for label in labels:
        if is_mmproj_name(label):
            mmprojs.append(label)
        else:
            models.append(label)
    return models, mmprojs


def do_list_models(
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
) -> tuple[list[str], list[str], str]:
    """モデル一覧を取得し、本体と mmproj に分ける。

    Returns
    -------
    models : list of str
        ドロップダウン用（本体）。
    mmprojs : list of str
        mmproj ドロップダウン用。
    message : str
        結果メッセージ。
    """
    cfg = cfg_from_inputs(
        inference_url, control_url, token, ngl, ctx, timeout_sec
    )
    try:
        models_info = make_control_client(cfg).list_models()
        if not models_info:
            return [], [], (
                "モデルがありません"
                "（カタログが空、またはサーバー未起動）。"
            )
        models: list[str] = []
        mmprojs: list[str] = []
        missing_n = 0
        for m in models_info:
            fname = str(m.extra.get("filename") or m.path or m.id)
            label = m.label()
            if m.extra.get("missing"):
                missing_n += 1
            if is_mmproj_name(fname) or is_mmproj_name(label):
                mmprojs.append(label)
            else:
                models.append(label)
        msg = (
            f"{len(models)} 件のモデル"
            f"（mmproj 候補 {len(mmprojs)} 件）を取得しました。"
        )
        if missing_n:
            msg += f" 欠落ファイル {missing_n} 件。"
        return models, mmprojs, msg
    except ControlClientError as e:
        return [], [], str(e)


def format_download_progress(prog: dict) -> str:
    """ダウンロード進捗をログ用文字列にする。

    Parameters
    ----------
    prog : dict
        制御APIの progress 応答。

    Returns
    -------
    str
        進捗テキスト。
    """
    label = str(prog.get("label") or prog.get("filename") or "ダウンロード")
    phase = str(prog.get("phase") or "")
    message = str(prog.get("message") or "")
    downloaded = int(prog.get("downloaded") or 0)
    total = prog.get("total")
    percent = prog.get("percent")

    def _mb(n: float) -> str:
        return f"{n / (1024 * 1024):.1f} MB"

    lines = [f"【進捗】{label}"]
    if message:
        lines.append(message)
    if isinstance(total, (int, float)) and total > 0 and percent is not None:
        bar_len = 24
        filled = max(0, min(bar_len, int(bar_len * float(percent) / 100.0)))
        bar = "#" * filled + "-" * (bar_len - filled)
        lines.append(f"[{bar}] {float(percent):.1f}%")
        lines.append(f"{_mb(downloaded)} / {_mb(float(total))}")
    else:
        lines.append(f"{_mb(downloaded)} 受信中（合計サイズ不明）")
    if phase:
        lines.append(f"phase: {phase}")
    return "\n".join(lines)


def do_download_model(
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
    model_id: str,
    repo_url: str,
    filename: str,
    mmproj_filename: str = "",
    overwrite: bool = False,
) -> tuple[list[str], list[str], str]:
    """モデルをダウンロードし、一覧を更新する。

    Returns
    -------
    models : list of str
        モデル Dropdown 用。
    mmprojs : list of str
        mmproj Dropdown 用。
    message : str
        結果ログ。
    """
    cfg = cfg_from_inputs(
        inference_url, control_url, token, ngl, ctx, timeout_sec
    )
    mid = (model_id or "").strip()
    url = (repo_url or "").strip()
    fname = (filename or "").strip()
    mmproj = (mmproj_filename or "").strip()
    if not mid or not url or not fname:
        return [], [], format_error(
            "タイトル（ID）・リポジトリ・LLMモデルをすべて入力してください。"
        )
    try:
        raw = make_control_client(cfg).download_model(
            mid,
            url,
            fname,
            mmproj_filename=mmproj or None,
            overwrite=bool(overwrite),
        )
        models, mmprojs, list_msg = do_list_models(
            inference_url, control_url, token, ngl, ctx, timeout_sec
        )
        detail = json.dumps(raw, ensure_ascii=False, indent=2)
        return models, mmprojs, f"ダウンロード完了\n{detail}\n{list_msg}"
    except ControlClientError as e:
        return [], [], str(e)


def do_delete_model(
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
    model_id: str,
) -> tuple[list[str], list[str], str]:
    """カタログからモデルを削除し、一覧を更新する。

    Returns
    -------
    models : list of str
        モデル Dropdown 用。
    mmprojs : list of str
        mmproj Dropdown 用。
    message : str
        結果ログ。
    """
    cfg = cfg_from_inputs(
        inference_url, control_url, token, ngl, ctx, timeout_sec
    )
    mid = (model_id or "").strip()
    if not mid:
        return [], [], format_error("削除するモデルを選択してください。")
    try:
        raw = make_control_client(cfg).delete_model(mid, delete_file=True)
        models, mmprojs, list_msg = do_list_models(
            inference_url, control_url, token, ngl, ctx, timeout_sec
        )
        detail = json.dumps(raw, ensure_ascii=False, indent=2)
        return models, mmprojs, f"削除完了\n{detail}\n{list_msg}"
    except ControlClientError as e:
        return [], [], str(e)


def ctrl_action(
    action: str,
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
    model: Optional[str] = None,
    mmproj: Optional[str] = None,
) -> str:
    """制御 API の操作を実行し結果文字列を返す。

    Parameters
    ----------
    action : str
        ``load`` / ``unload`` / ``stop`` / ``restart`` / ``start``。
    model : str or None
        load / start で使うモデル。
    mmproj : str or None
        任意のプロジェクター。ネイティブ Vision では空。

    Returns
    -------
    str
        結果またはエラー文。
    """
    cfg = cfg_from_inputs(
        inference_url, control_url, token, ngl, ctx, timeout_sec
    )
    client = make_control_client(cfg)
    mmproj_val = (mmproj or "").strip() or None
    try:
        if action == "load":
            if not model:
                return format_error("ロード前にモデルを選択してください。")
            raw = client.load(
                model, ngl=cfg.ngl, ctx=cfg.ctx, mmproj=mmproj_val
            )
        elif action == "unload":
            raw = client.unload()
        elif action == "stop":
            raw = client.stop()
        elif action == "restart":
            raw = client.restart()
        elif action == "start":
            if model:
                raw = client.start(
                    model=model, ngl=cfg.ngl, ctx=cfg.ctx, mmproj=mmproj_val
                )
            else:
                raw = client.start()
        else:
            return f"不明な操作: {action}"
        return f"{action} ok\n{json.dumps(raw, ensure_ascii=False, indent=2)}"
    except ControlClientError as e:
        return str(e)


def history_to_messages(
    history: list,
    *,
    max_long_edge: int = 1024,
) -> list[ChatMessage]:
    """Gradio Chatbot 履歴をメッセージ列に変換する。

    Parameters
    ----------
    history : list
        ``[[user, assistant], ...]`` 形式。ユーザ側は文字列または
        ``(画像パス, キャプション)``。
    max_long_edge : int, default 1024
        画像の長い辺上限（送信時縮小）。

    Returns
    -------
    list of ChatMessage
        API 送信用メッセージ。
    """
    messages: list[ChatMessage] = []
    for pair in history or []:
        if not pair:
            continue
        user = pair[0]
        assistant = pair[1] if len(pair) > 1 else None
        if user:
            text, images = extract_user_turn(user)
            try:
                content = build_user_content(
                    text, images, max_long_edge=int(max_long_edge)
                )
            except (OSError, ValueError) as e:
                content = text or str(e)
            if content:
                messages.append(ChatMessage(role="user", content=content))
        if assistant:
            messages.append(
                ChatMessage(role="assistant", content=str(assistant))
            )
    return messages


def _parse_chat_input(message: ChatInput) -> tuple[str, list[str]]:
    """MultimodalTextbox / Textbox 入力を正規化する。

    Parameters
    ----------
    message : str or dict or None
        UI 入力。

    Returns
    -------
    text : str
        文言。
    files : list of str
        画像パス。
    """
    if message is None:
        return "", []
    if isinstance(message, dict):
        text = str(message.get("text") or "").strip()
        files_raw = message.get("files") or []
        files: list[str] = []
        for f in files_raw:
            if isinstance(f, dict):
                path = f.get("path") or f.get("name") or ""
                if path:
                    files.append(str(path))
            elif f:
                files.append(str(f))
        return text, files
    return str(message).strip(), []


def format_chat_metrics(
    *,
    vram_used_gb: float | None = None,
    vram_total_gb: float | None = None,
    tok_s: float | None = None,
) -> str:
    """Chat 用メトリクス 1 行を組み立てる。

    Parameters
    ----------
    vram_used_gb : float or None
        使用中 VRAM（GB）。
    vram_total_gb : float or None
        総 VRAM（GB）。
    tok_s : float or None
        生成速度。

    Returns
    -------
    str
        例: ``VRAM 8.2 / 24.0 GB · 42.3 tok/s``
    """
    if vram_used_gb is not None and vram_total_gb is not None:
        vram = f"VRAM {vram_used_gb:.1f} / {vram_total_gb:.1f} GB"
    else:
        vram = "VRAM —"
    if tok_s is not None:
        speed = f"{tok_s:.1f} tok/s"
    else:
        speed = "— tok/s"
    return f"{vram} · {speed}"


def chat_respond(
    message: ChatInput,
    history: list,
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
    temperature: float,
    image_max_long_edge: float | int = 1024,
) -> Generator[tuple[list, str], None, None]:
    """チャット送信に応答し Chatbot 履歴とメトリクスを逐次 yield する。

    Yields
    ------
    history : list
        Gradio Chatbot 用履歴。
    metrics : str
        VRAM / tok/s 表示行。
    """
    history = list(history or [])
    metrics = format_chat_metrics()
    text, files = _parse_chat_input(message)
    if not text and not files:
        yield history, metrics
        return

    max_edge = int(image_max_long_edge)
    try:
        display = chatbot_user_display(text, files)
        api_content = build_user_content(
            text, files, max_long_edge=max_edge
        )
    except (OSError, ValueError) as e:
        history = history + [[text or "(画像)", format_error(str(e))]]
        yield history, metrics
        return

    if not api_content:
        yield history, metrics
        return

    history = history + [[display, None]]
    yield history, metrics

    cfg = cfg_from_inputs(
        inference_url, control_url, token, ngl, ctx, timeout_sec
    )
    try:
        st = make_control_client(cfg).status()
        if not st.llama_running:
            history[-1][1] = format_error(
                "モデル未ロードです（llama_running=false）。"
                "Connection からロードしてください。"
            )
            yield history, format_chat_metrics(
                vram_used_gb=st.vram_used_gb,
                vram_total_gb=st.vram_total_gb,
            )
            return
        metrics = format_chat_metrics(
            vram_used_gb=st.vram_used_gb,
            vram_total_gb=st.vram_total_gb,
        )
    except ControlClientError:
        pass

    messages = history_to_messages(history[:-1], max_long_edge=max_edge)
    messages.append(ChatMessage(role="user", content=api_content))
    backend = make_http_backend(cfg)
    acc = ""
    tok_s: float | None = None
    try:
        for chunk in backend.chat_stream(
            messages, temperature=float(temperature)
        ):
            if chunk.tok_s is not None:
                tok_s = chunk.tok_s
            if chunk.text:
                acc += chunk.text
                history[-1][1] = acc
                yield history, metrics
            if chunk.done:
                break
        if history[-1][1] is None:
            history[-1][1] = "（空の応答）"
    except HttpBackendError as e:
        history[-1][1] = str(e)

    used = total = None
    try:
        st2 = make_control_client(cfg).status()
        used, total = st2.vram_used_gb, st2.vram_total_gb
    except ControlClientError:
        pass
    yield history, format_chat_metrics(
        vram_used_gb=used,
        vram_total_gb=total,
        tok_s=tok_s,
    )


def refresh_models(
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
):
    """モデル／mmproj Dropdown を更新する。

    Returns
    -------
    update
        モデル Dropdown。
    update
        mmproj Dropdown。
    str
        ログメッセージ。
    """
    models, mmprojs, msg = do_list_models(
        inference_url, control_url, token, ngl, ctx, timeout_sec
    )
    mmproj_choices = [""] + mmprojs
    return (
        gr.update(choices=models, value=(models[0] if models else None)),
        gr.update(choices=mmproj_choices, value=""),
        msg,
    )


def download_and_refresh(
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
    model_id: str,
    repo_url: str,
    filename: str,
    mmproj_filename: str,
    overwrite: bool,
):
    """ダウンロード中は進捗をログに出し、完了後に Dropdown を更新する。

    Yields
    ------
    update
        モデル Dropdown。
    update
        mmproj Dropdown。
    str
        ログ（進捗または結果）。
    """
    mid = (model_id or "").strip()
    url = (repo_url or "").strip()
    fname = (filename or "").strip()
    mmproj = (mmproj_filename or "").strip()
    if not mid or not url or not fname:
        yield (
            gr.update(),
            gr.update(),
            format_error(
                "タイトル（ID）・リポジトリ・LLMモデルをすべて入力してください。"
            ),
        )
        return

    cfg = cfg_from_inputs(
        inference_url, control_url, token, ngl, ctx, timeout_sec
    )
    client = make_control_client(cfg)
    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result["raw"] = client.download_model(
                mid,
                url,
                fname,
                mmproj_filename=mmproj or None,
                overwrite=bool(overwrite),
            )
        except BaseException as e:  # noqa: BLE001
            error["e"] = e

    yield gr.update(), gr.update(), "ダウンロードを開始しました…"
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while thread.is_alive():
        try:
            prog = client.download_progress()
            yield gr.update(), gr.update(), format_download_progress(prog)
        except ControlClientError as e:
            yield gr.update(), gr.update(), f"進捗取得: {e}"
        time.sleep(0.5)
    thread.join(timeout=1)

    if error:
        err = error["e"]
        msg = str(err) if isinstance(err, ControlClientError) else format_error(str(err))
        try:
            prog = client.download_progress()
            msg = f"{format_download_progress(prog)}\n\n{msg}"
        except ControlClientError:
            pass
        yield gr.update(), gr.update(), msg
        return

    models, mmprojs, list_msg = do_list_models(
        inference_url, control_url, token, ngl, ctx, timeout_sec
    )
    detail = json.dumps(result.get("raw") or {}, ensure_ascii=False, indent=2)
    msg = f"ダウンロード完了\n{detail}\n{list_msg}"
    mmproj_choices = [""] + mmprojs
    yield (
        gr.update(choices=models, value=(models[0] if models else None)),
        gr.update(choices=mmproj_choices, value=""),
        msg,
    )


def delete_and_refresh(
    inference_url: str,
    control_url: str,
    token: str,
    ngl: float | int,
    ctx: float | int,
    timeout_sec: float,
    model_id: str,
):
    """削除後に Dropdown を更新する。

    Returns
    -------
    update
        モデル Dropdown。
    update
        mmproj Dropdown。
    str
        ログ。
    """
    models, mmprojs, msg = do_delete_model(
        inference_url,
        control_url,
        token,
        ngl,
        ctx,
        timeout_sec,
        model_id,
    )
    if "削除完了" not in msg:
        return gr.update(), gr.update(), msg
    mmproj_choices = [""] + mmprojs
    return (
        gr.update(choices=models, value=(models[0] if models else None)),
        gr.update(choices=mmproj_choices, value=""),
        msg,
    )


def empty_multimodal() -> dict[str, Any]:
    """MultimodalTextbox を空にする値を返す。

    Returns
    -------
    dict
        ``{"text": "", "files": []}``。
    """
    return {"text": "", "files": []}
