"""OpenAI 互換の推論クライアント（llama-server 向け）。"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional, Sequence

import httpx

from core.errors import format_error
from core.types import ChatChunk, ChatMessage


class HttpBackendError(Exception):
    """推論 HTTP 呼び出しの失敗。

    Parameters
    ----------
    message : str
        エラー内容。
    timeout : bool, default False
        タイムアウト由来なら True。
    """

    def __init__(self, message: str, *, timeout: bool = False) -> None:
        super().__init__(format_error(message, timeout=timeout))
        self.timeout = timeout


class HttpBackend:
    """llama-server へのチャット／ヘルスチェッククライアント。

    Parameters
    ----------
    base_url : str
        チャット用 URL（llama-server）。
    timeout_sec : float, default 60.0
        接続・通常リクエストのタイムアウト秒。
    stream_timeout_sec : float, default 300.0
        ストリーム全体のタイムアウト秒。
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_sec: float = 60.0,
        stream_timeout_sec: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.stream_timeout_sec = stream_timeout_sec

    def health(self) -> bool:
        """推論サーバーの疎通を確認する。

        Returns
        -------
        bool
            応答が 5xx 未満なら True。

        Raises
        ------
        HttpBackendError
            タイムアウトまたは接続失敗。
        """
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                r = client.get(url)
                if r.status_code < 500:
                    return True
        except httpx.TimeoutException as e:
            raise HttpBackendError(
                str(e) or "health タイムアウト", timeout=True
            ) from e
        except httpx.HTTPError as e:
            try:
                with httpx.Client(timeout=self.timeout_sec) as client:
                    r = client.get(f"{self.base_url}/v1/models")
                    return r.status_code < 500
            except httpx.TimeoutException as e2:
                raise HttpBackendError(
                    str(e2) or "health タイムアウト", timeout=True
                ) from e2
            except httpx.HTTPError:
                raise HttpBackendError(str(e)) from e
        return False

    def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[ChatChunk]:
        """チャット補完をストリームで取得する。

        Parameters
        ----------
        messages : sequence of ChatMessage
            会話履歴。``content`` は文字列または multimodal パーツ列。
        model : str, default "default"
            モデル名（サーバー側のエイリアス）。
        temperature : float, default 0.7
            温度。
        max_tokens : int or None, default None
            最大生成トークン。
        top_p : float or None, default None
            nucleus sampling。

        Yields
        ------
        ChatChunk
            テキスト断片または終了印。

        Raises
        ------
        HttpBackendError
            HTTP エラーまたはタイムアウト。
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "stream": True,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p

        timeout = httpx.Timeout(
            self.stream_timeout_sec, connect=self.timeout_sec
        )
        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code >= 400:
                        body = resp.read().decode("utf-8", errors="replace")
                        raise HttpBackendError(
                            f"HTTP {resp.status_code}: "
                            f"{body or resp.reason_phrase}"
                        )
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data = line[5:].strip()
                        else:
                            data = line.strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            yield ChatChunk(text="", done=True)
                            return
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content") or ""
                        if piece:
                            yield ChatChunk(text=piece, done=False)
                        finish = choices[0].get("finish_reason")
                        if finish:
                            yield ChatChunk(text="", done=True)
                            return
        except httpx.TimeoutException as e:
            raise HttpBackendError(
                str(e) or "stream タイムアウト", timeout=True
            ) from e
        except HttpBackendError:
            raise
        except httpx.HTTPError as e:
            raise HttpBackendError(str(e)) from e
