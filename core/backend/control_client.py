"""制御 API クライアント。

サーバーとの契約:

- ヘッダ ``X-Control-Token``
- ``GET  /v1/control/status``
- ``GET  /v1/control/models``
- ``POST /v1/control/models/download`` （body: id, url, filename, 任意 mmproj_filename）
- ``POST /v1/control/models/delete`` （body: id, delete_file）
- ``POST /v1/control/load`` （body: model, ngl, ctx, 任意 mmproj）
- ``POST /v1/control/unload``
- ``POST /v1/control/restart``
- ``POST /v1/control/start`` / ``stop`` （load / unload の別名可）
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from core.errors import format_error
from core.types import ModelInfo, ServerStatus


class ControlClientError(Exception):
    """制御 API 呼び出しの失敗。

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


class ControlClient:
    """サーバー制御 API のクライアント。

    Parameters
    ----------
    base_url : str
        モデル操作 URL（制御API）。
    token : str, default ""
        共有トークン。
    timeout_sec : float, default 60.0
        タイムアウト秒。
    """

    def __init__(
        self,
        base_url: str,
        token: str = "",
        *,
        timeout_sec: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_sec = timeout_sec

    def _headers(self) -> dict[str, str]:
        """認証付き HTTP ヘッダを返す。"""
        h = {"Accept": "application/json"}
        if self.token:
            h["X-Control-Token"] = self.token
        return h

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        timeout_sec: Optional[float] = None,
    ) -> dict[str, Any]:
        """制御 API に 1 リクエスト送る。

        Parameters
        ----------
        method : str
            HTTP メソッド。
        path : str
            ``/v1/control/...`` 形式のパス。
        json_body : dict or None
            JSON ボディ。
        timeout_sec : float or None
            省略時はクライアント既定。

        Returns
        -------
        dict
            応答 JSON。

        Raises
        ------
        ControlClientError
            認証失敗・HTTP エラー・タイムアウト。
        """
        url = f"{self.base_url}{path}"
        timeout = self.timeout_sec if timeout_sec is None else timeout_sec
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.request(
                    method, url, headers=self._headers(), json=json_body
                )
                text = r.text
                if r.status_code == 401:
                    raise ControlClientError(
                        "認証エラー（制御トークンを確認してください）"
                    )
                if r.status_code >= 400:
                    detail = text or r.reason_phrase
                    try:
                        err_obj = r.json()
                        if isinstance(err_obj, dict) and err_obj.get("error"):
                            detail = str(err_obj["error"])
                    except ValueError:
                        pass
                    raise ControlClientError(
                        f"HTTP {r.status_code}: {detail}"
                    )
                if not text.strip():
                    return {}
                try:
                    data = r.json()
                except ValueError as e:
                    raise ControlClientError(
                        f"不正な JSON: {text[:500]}"
                    ) from e
                if not isinstance(data, dict):
                    return {"data": data}
                return data
        except httpx.TimeoutException as e:
            raise ControlClientError(
                str(e) or "制御 API タイムアウト", timeout=True
            ) from e
        except ControlClientError:
            raise
        except httpx.HTTPError as e:
            raise ControlClientError(str(e)) from e

    def status(self) -> ServerStatus:
        """制御・llama-server の状態を取得する。

        Returns
        -------
        ServerStatus
            状態。
        """
        raw = self._request("GET", "/v1/control/status")
        return ServerStatus(
            ok=bool(raw.get("ok", True)),
            llama_running=bool(
                raw.get("llama_running", raw.get("running", False))
            ),
            loaded_model=raw.get("loaded_model") or raw.get("model"),
            exit_code=raw.get("exit_code"),
            log_tail=str(raw.get("log_tail") or raw.get("log") or ""),
            raw=raw,
            message=str(raw.get("message") or ""),
        )

    def list_models(self) -> list[ModelInfo]:
        """カタログ（models.json）上のモデル一覧を取得する。

        Returns
        -------
        list of ModelInfo
            モデル一覧。``extra`` に ``url`` / ``filename`` / ``missing`` 等。
        """
        raw = self._request("GET", "/v1/control/models")
        items = raw.get("models") or raw.get("items") or []
        out: list[ModelInfo] = []
        for it in items:
            if isinstance(it, str):
                out.append(ModelInfo(id=it, name=it))
            elif isinstance(it, dict):
                mid = str(
                    it.get("id")
                    or it.get("name")
                    or it.get("filename")
                    or it.get("path")
                    or ""
                )
                if not mid:
                    continue
                fname = str(it.get("filename") or "")
                out.append(
                    ModelInfo(
                        id=mid,
                        name=str(it.get("name") or mid),
                        path=str(it.get("path") or fname),
                        extra={
                            k: v
                            for k, v in it.items()
                            if k not in ("id", "name", "path")
                        },
                    )
                )
        return out

    def download_model(
        self,
        model_id: str,
        url: str,
        filename: str,
        *,
        mmproj_filename: Optional[str] = None,
        overwrite: bool = False,
        timeout_sec: Optional[float] = None,
    ) -> dict[str, Any]:
        """モデルをダウンロードしてカタログに登録する。

        Parameters
        ----------
        model_id : str
            タイトル（ID）。
        url : str
            リポジトリ（URL または Hugging Face ``org/repo``）。
        filename : str
            LLM モデルの保存ファイル名。
        mmproj_filename : str or None, default None
            Vision（mmproj）ファイル名。空なら取得しない。
        overwrite : bool, default False
            同一 id / 同名ファイルの上書き。
        timeout_sec : float or None
            省略時は ``max(クライアント既定, 3600)``。

        Returns
        -------
        dict
            サーバー応答。
        """
        to = timeout_sec
        if to is None:
            to = max(float(self.timeout_sec), 3600.0)
        body: dict[str, Any] = {
            "id": model_id,
            "url": url,
            "filename": filename,
            "overwrite": overwrite,
        }
        mm = (mmproj_filename or "").strip()
        if mm:
            body["mmproj_filename"] = mm
        return self._request(
            "POST",
            "/v1/control/models/download",
            json_body=body,
            timeout_sec=to,
        )

    def delete_model(
        self,
        model_id: str,
        *,
        delete_file: bool = True,
    ) -> dict[str, Any]:
        """カタログからモデルを削除する（任意で実ファイルも）。

        Parameters
        ----------
        model_id : str
            削除するエントリ ID。
        delete_file : bool, default True
            実ファイルも削除するか。

        Returns
        -------
        dict
            サーバー応答。
        """
        return self._request(
            "POST",
            "/v1/control/models/delete",
            json_body={"id": model_id, "delete_file": delete_file},
        )

    def load(
        self,
        model: str,
        *,
        ngl: int = 99,
        ctx: int = 8192,
        mmproj: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """モデルをロードする（llama-server 起動）。

        Parameters
        ----------
        model : str
            モデル ID またはパス。
        ngl : int, default 99
            GPU 層数。
        ctx : int, default 8192
            コンテキスト長。
        mmproj : str or None, default None
            別ファイルの Vision プロジェクター。省略時は付けない
            （テキスト専用、または Vision ネイティブ統合 GGUF）。
        extra : dict or None
            追加パラメータ。

        Returns
        -------
        dict
            サーバー応答。
        """
        body: dict[str, Any] = {"model": model, "ngl": ngl, "ctx": ctx}
        if mmproj:
            body["mmproj"] = mmproj
        if extra:
            body.update(extra)
        return self._request("POST", "/v1/control/load", json_body=body)

    def unload(self) -> dict[str, Any]:
        """モデルをアンロードする（プロセス停止）。

        Returns
        -------
        dict
            サーバー応答。
        """
        return self._request("POST", "/v1/control/unload", json_body={})

    def restart(self) -> dict[str, Any]:
        """同一モデルで再起動する。

        Returns
        -------
        dict
            サーバー応答。
        """
        return self._request("POST", "/v1/control/restart", json_body={})

    def start(self, **kwargs: Any) -> dict[str, Any]:
        """起動する。

        ``model`` があれば :meth:`load` に委譲する。

        Returns
        -------
        dict
            サーバー応答。
        """
        model = kwargs.pop("model", None)
        if model:
            return self.load(str(model), **kwargs)
        return self._request("POST", "/v1/control/start", json_body=kwargs or {})

    def stop(self) -> dict[str, Any]:
        """停止する（unload 相当の別名）。

        Returns
        -------
        dict
            サーバー応答。
        """
        return self._request("POST", "/v1/control/stop", json_body={})
