"""画像付きメッセージの OpenAI 互換 content 組み立て。"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Optional, Sequence, Union

ContentPart = dict[str, Any]
MessageContent = Union[str, list[ContentPart]]

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def is_mmproj_name(name: str) -> bool:
    """ファイル名が mmproj（プロジェクター）らしいか判定する。

    Parameters
    ----------
    name : str
        モデル ID またはファイル名。

    Returns
    -------
    bool
        mmproj らしければ True。
    """
    lower = Path(name).name.lower()
    return "mmproj" in lower


def is_image_path(path: str) -> bool:
    """パスが画像ファイルらしいか判定する。

    Parameters
    ----------
    path : str
        ファイルパス。

    Returns
    -------
    bool
        画像拡張子なら True。
    """
    return Path(path).suffix.lower() in _IMAGE_SUFFIXES


def file_to_data_url(path: str) -> str:
    """画像ファイルを data URL（base64）に変換する。

    Parameters
    ----------
    path : str
        ローカル画像パス。

    Returns
    -------
    str
        ``data:<mime>;base64,...`` 形式。

    Raises
    ------
    OSError
        読み込み失敗。
    ValueError
        空ファイルなど。
    """
    p = Path(path)
    raw = p.read_bytes()
    if not raw:
        raise ValueError(f"空の画像ファイルです: {p}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_user_content(
    text: str,
    image_paths: Optional[Sequence[str]] = None,
) -> MessageContent:
    """テキストと画像パスから OpenAI 互換 content を組み立てる。

    画像が無ければ文字列のみ返す。ネイティブ Vision GGUF も
    mmproj 付き VLM も、推論口では同じ multimodal content を使う。

    Parameters
    ----------
    text : str
        ユーザ文言。
    image_paths : sequence of str or None
        画像ファイルパス。

    Returns
    -------
    str or list of dict
        文字列、または text / image_url パーツのリスト。
    """
    paths = [p for p in (image_paths or []) if p]
    text = (text or "").strip()
    if not paths:
        return text

    parts: list[ContentPart] = []
    if text:
        parts.append({"type": "text", "text": text})
    for path in paths:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": file_to_data_url(path)},
            }
        )
    return parts


def chatbot_user_display(
    text: str,
    image_paths: Optional[Sequence[str]] = None,
) -> Any:
    """Gradio Chatbot（tuples）向けのユーザ表示値を作る。

    Parameters
    ----------
    text : str
        文言。
    image_paths : sequence of str or None
        画像パス。

    Returns
    -------
    str or tuple
        テキストのみ、または ``(先頭画像パス, キャプション)``。
    """
    paths = [p for p in (image_paths or []) if p]
    text = (text or "").strip()
    if not paths:
        return text
    caption = text
    if len(paths) > 1:
        extra = f"（他 {len(paths) - 1} 枚）"
        caption = f"{caption} {extra}".strip() if caption else extra
    return (paths[0], caption) if caption else (paths[0],)


def extract_user_turn(user: Any) -> tuple[str, list[str]]:
    """Chatbot 履歴のユーザ側からテキストと画像パスを取り出す。

    Parameters
    ----------
    user : any
        文字列、または ``(path, alt)`` / ``(path,)``。

    Returns
    -------
    text : str
        文言。
    image_paths : list of str
        画像パス（0 件以上）。
    """
    if user is None:
        return "", []
    if isinstance(user, (tuple, list)) and user:
        first = user[0]
        if isinstance(first, str) and (
            is_image_path(first) or Path(first).is_file()
        ):
            alt = ""
            if len(user) > 1 and user[1] is not None:
                alt = str(user[1])
            return alt.strip(), [first]
        return str(user), []
    return str(user).strip(), []
