"""エラー整形と VRAM 不足時のユーザ向けヒント（方針 A+B）。"""

from __future__ import annotations

import re

_OOM_PATTERNS = re.compile(
    r"out of memory|oom|cuda.*(error|oom)|failed to allocate|insufficient memory|vram",
    re.IGNORECASE,
)

VRAM_HINT = (
    "VRAM / メモリ不足の可能性があります。"
    "対策例: ngl を下げる、ctx を下げる、より小さい量子化、他 GPU プロセスを終了。"
    "Vision（画像）利用時は画像サイズや枚数も減らしてください。"
    "その後 Connection から unload / restart / load を実行してください。"
)

TIMEOUT_HINT = (
    "サーバーが時間内に応答しませんでした。"
    "status を確認し、Connection から restart または unload→load を行ってください"
    "（無限待ちしないでください）。"
)

CONNECTION_HINT = (
    "モデル操作 URL（制御API）に接続できません。"
    "setup_server で制御APIを起動しているか、"
    "URL（例: http://127.0.0.1:8090）とトークンを確認してください。"
)

_CONN_PATTERNS = re.compile(
    r"10061|connection refused|connecterror|actively refused|"
    r"接続できませんでした|対象のコンピューターによって拒否|"
    r"failed to establish|name or service not known|getaddrinfo failed|"
    r"nodename nor servname|network is unreachable",
    re.IGNORECASE,
)


def looks_like_oom(text: str) -> bool:
    """メッセージが OOM / VRAM 不足らしいか判定する。

    Parameters
    ----------
    text : str
        エラー文字列。

    Returns
    -------
    bool
        該当しそうなら True。
    """
    return bool(text and _OOM_PATTERNS.search(text))


def looks_like_connection_error(text: str) -> bool:
    """接続拒否・到達不能らしいか判定する。

    Parameters
    ----------
    text : str
        エラー文字列。

    Returns
    -------
    bool
        該当しそうなら True。
    """
    return bool(text and _CONN_PATTERNS.search(text))


def format_error(message: str, *, timeout: bool = False) -> str:
    """ユーザ向けエラー文にヒントを付与する。

    Parameters
    ----------
    message : str
        元メッセージ。
    timeout : bool, default False
        タイムアウト由来なら True。

    Returns
    -------
    str
        表示用文字列。
    """
    msg = (message or "不明なエラー").strip()
    parts = [msg]
    if timeout:
        parts.append(TIMEOUT_HINT)
    elif looks_like_oom(msg):
        parts.append(VRAM_HINT)
    elif looks_like_connection_error(msg):
        parts.append(CONNECTION_HINT)
    return "\n\n".join(parts)
