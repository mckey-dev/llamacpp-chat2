"""共有型定義（Gradio 非依存）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

# OpenAI 互換: 文字列、または multimodal パーツ列（text / image_url）。
MessageContent = Union[str, list[dict[str, Any]]]


@dataclass
class ChatMessage:
    """チャット 1 メッセージ。

    Attributes
    ----------
    role : str
        役割（``user`` / ``assistant`` / ``system`` など）。
    content : str or list of dict
        本文。Vision 利用時は OpenAI 形式のパーツ列
        （``{"type":"text",...}`` / ``{"type":"image_url",...}``）。
    """

    role: str
    content: MessageContent


@dataclass
class ChatChunk:
    """ストリーミング応答の断片。

    Attributes
    ----------
    text : str
        追加テキスト。
    done : bool, default False
        ストリーム終了なら True。
    tok_s : float or None, default None
        生成速度（tokens/sec）。終了チャンクで付くことがある。
    """

    text: str
    done: bool = False
    tok_s: Optional[float] = None


@dataclass
class ModelInfo:
    """サーバー上のモデル情報。

    Attributes
    ----------
    id : str
        モデル識別子。
    name : str, default ""
        表示名。空なら ``id`` を用いる。
    path : str, default ""
        ファイルパス（分かる場合）。
    extra : dict
        その他の属性。
    """

    id: str
    name: str = ""
    path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        """UI 表示用ラベルを返す。

        Returns
        -------
        str
            ``name`` があればそれ、なければ ``id``。
        """
        return self.name or self.id


@dataclass
class ServerStatus:
    """制御 API の status 応答。

    Attributes
    ----------
    ok : bool
        制御 API として正常か。
    llama_running : bool, default False
        ``llama-server`` が動作中か。
    loaded_model : str or None
        ロード中モデル。
    exit_code : int or None
        直近終了コード。
    log_tail : str
        ログ末尾。
    raw : dict
        生 JSON。
    message : str
        付加メッセージ。
    vram_used_gb : float or None
        使用中 VRAM（GB）。
    vram_total_gb : float or None
        総 VRAM（GB）。
    """

    ok: bool
    llama_running: bool = False
    loaded_model: Optional[str] = None
    exit_code: Optional[int] = None
    log_tail: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None


@dataclass
class ConnectionConfig:
    """フロントエンドの接続・ロード設定。

    Attributes
    ----------
    inference_base_url : str
        チャット用 URL（llama-server）。
    control_base_url : str
        モデル操作 URL（制御API）。
    control_token : str
        制御APIトークン。既定は ``llamacpp-chat2``
        （サーバーの ``LLAMACPP_CHAT2_CONTROL_TOKEN`` と同じ値にする）。
    ngl : int
        ``--n-gpu-layers``。
    ctx : int
        ``--ctx-size``。
    timeout_sec : float
        通常リクエストのタイムアウト秒。
    stream_timeout_sec : float
        ストリーム全体のタイムアウト秒。
    image_max_long_edge : int
        チャット送信時の画像長い辺上限（px）。0 以下で無制限。
    """

    inference_base_url: str = "http://127.0.0.1:8080"
    control_base_url: str = "http://127.0.0.1:8090"
    control_token: str = "llamacpp-chat2"
    ngl: int = 99
    ctx: int = 8192
    timeout_sec: float = 60.0
    stream_timeout_sec: float = 300.0
    image_max_long_edge: int = 1024
