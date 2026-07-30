"""制御 API（http.server・標準ライブラリのみ）。

エンドポイント:

- ``GET  /v1/control/status``
- ``GET  /v1/control/models``
- ``POST /v1/control/models/download``
- ``POST /v1/control/load`` / ``unload`` / ``restart`` / ``start`` / ``stop``
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from server.models_catalog import (
    CatalogError,
    default_models_dir,
    download_model,
    list_models_with_status,
)
from server.process_manager import (
    ProcessManagerError,
    default_process_manager,
)

DEFAULT_TOKEN = "llamacpp-chat2"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8090


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class ControlState:
    """制御 API の共有状態。"""

    def __init__(
        self,
        *,
        token: str,
        models_dir: Path,
        process_manager: Any,
    ) -> None:
        self.token = token
        self.models_dir = models_dir
        self.process_manager = process_manager


def make_handler(state: ControlState):
    """RequestHandler クラスを生成する。"""

    class Handler(BaseHTTPRequestHandler):
        """制御 API ハンドラ。"""

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                data = json.loads(raw.decode("utf-8"))
            except ValueError as e:
                raise CatalogError(f"不正な JSON: {e}", status=400) from e
            if not isinstance(data, dict):
                raise CatalogError("JSON オブジェクトが必要です", status=400)
            return data

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _check_token(self) -> bool:
            got = self.headers.get("X-Control-Token") or ""
            auth = self.headers.get("Authorization") or ""
            if auth.lower().startswith("bearer "):
                got = auth[7:].strip() or got
            if got != state.token:
                self._send_json(
                    401, {"ok": False, "error": "unauthorized"}
                )
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802
            if not self._check_token():
                return
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path == "/v1/control/status":
                    body = state.process_manager.status_payload()
                    body["models_dir"] = str(state.models_dir)
                    self._send_json(200, body)
                    return
                if path == "/v1/control/models":
                    models = list_models_with_status(state.models_dir)
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "models": models,
                            "models_dir": str(state.models_dir),
                        },
                    )
                    return
                self._send_json(404, {"ok": False, "error": "not found"})
            except CatalogError as e:
                self._send_json(
                    e.status, {"ok": False, "error": str(e)}
                )
            except ProcessManagerError as e:
                self._send_json(
                    e.status, {"ok": False, "error": str(e)}
                )
            except Exception as e:  # noqa: BLE001
                self._send_json(
                    500, {"ok": False, "error": f"internal: {e}"}
                )

        def do_POST(self) -> None:  # noqa: N802
            if not self._check_token():
                return
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                body = self._read_json()
                if path == "/v1/control/models/download":
                    result = download_model(
                        state.models_dir,
                        model_id=str(body.get("id") or ""),
                        url=str(body.get("url") or ""),
                        filename=str(body.get("filename") or ""),
                        overwrite=bool(body.get("overwrite", False)),
                    )
                    self._send_json(200, result)
                    return
                if path == "/v1/control/load":
                    model = str(body.get("model") or "")
                    result = state.process_manager.load(
                        model,
                        ngl=int(body.get("ngl", 99)),
                        ctx=int(body.get("ctx", 8192)),
                        mmproj=str(body.get("mmproj") or "").strip() or None,
                    )
                    self._send_json(200, result)
                    return
                if path == "/v1/control/unload":
                    result = state.process_manager.unload()
                    self._send_json(200, result)
                    return
                if path == "/v1/control/restart":
                    result = state.process_manager.restart()
                    self._send_json(200, result)
                    return
                if path == "/v1/control/start":
                    kwargs: dict[str, Any] = {}
                    if body.get("model"):
                        kwargs["model"] = str(body.get("model"))
                    if "ngl" in body:
                        kwargs["ngl"] = int(body.get("ngl"))
                    if "ctx" in body:
                        kwargs["ctx"] = int(body.get("ctx"))
                    if body.get("mmproj"):
                        kwargs["mmproj"] = str(body.get("mmproj"))
                    result = state.process_manager.start(**kwargs)
                    self._send_json(200, result)
                    return
                if path == "/v1/control/stop":
                    result = state.process_manager.stop()
                    self._send_json(200, result)
                    return
                self._send_json(404, {"ok": False, "error": "not found"})
            except CatalogError as e:
                self._send_json(
                    e.status, {"ok": False, "error": str(e)}
                )
            except ProcessManagerError as e:
                self._send_json(
                    e.status, {"ok": False, "error": str(e)}
                )
            except Exception as e:  # noqa: BLE001
                self._send_json(
                    500, {"ok": False, "error": f"internal: {e}"}
                )

    return Handler


def run(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    token: Optional[str] = None,
    models_dir: Optional[Path] = None,
) -> None:
    """制御 API を起動する。"""
    host = host or _env("LLAMACPP_CHAT2_CONTROL_HOST", DEFAULT_HOST)
    port = port or int(
        _env("LLAMACPP_CHAT2_CONTROL_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT
    )
    token = token or _env("LLAMACPP_CHAT2_CONTROL_TOKEN", DEFAULT_TOKEN) or DEFAULT_TOKEN
    models_dir = models_dir or default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    pm = default_process_manager(models_dir)
    state = ControlState(token=token, models_dir=models_dir, process_manager=pm)
    handler = make_handler(state)
    server = ThreadingHTTPServer((host, port), handler)
    print(
        f"control API listening on http://{host}:{port}\n"
        f"models_dir={models_dir}\n"
        f"inference=http://{pm.llama_host}:{pm.llama_port}\n"
        f"token={'*' * max(len(token), 1)}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
        pm.unload()
        server.shutdown()


def main() -> None:
    """エントリポイント。"""
    # リポジトリルートを path に入れる（python -m server.control_api）
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    run()


if __name__ == "__main__":
    main()
