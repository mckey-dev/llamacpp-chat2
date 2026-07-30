"""llama-server 子プロセスの起動・停止・状態管理。

標準ライブラリのみ。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from server.models_catalog import CatalogError, load_catalog


class ProcessManagerError(Exception):
    """子プロセス操作の失敗。"""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


DEFAULT_LLAMA_HOST = "127.0.0.1"
DEFAULT_LLAMA_PORT = 8080
LOG_TAIL_MAX_CHARS = 32_000
STARTUP_WAIT_SEC = 2.0
TERMINATE_TIMEOUT_SEC = 15.0


@dataclass
class LoadParams:
    """直近 load のパラメータ（restart 用）。"""

    model: str
    ngl: int = 99
    ctx: int = 8192
    mmproj: Optional[str] = None
    host: str = DEFAULT_LLAMA_HOST
    port: int = DEFAULT_LLAMA_PORT


@dataclass
class ProcessManager:
    """llama-server 子プロセスを 1 本だけ管理する。

    Parameters
    ----------
    models_dir : pathlib.Path
        モデル一時ディレクトリ。
    llama_server : str or None, default None
        実行ファイルパス。省略時は環境変数または PATH を探す。
    llama_host : str, default "127.0.0.1"
        バインドホスト。
    llama_port : int, default 8080
        推論ポート。
    """

    models_dir: Path
    llama_server: Optional[str] = None
    llama_host: str = DEFAULT_LLAMA_HOST
    llama_port: int = DEFAULT_LLAMA_PORT

    _proc: Optional[subprocess.Popen[str]] = field(
        default=None, init=False, repr=False
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _log_lines: deque[str] = field(
        default_factory=deque, init=False, repr=False
    )
    _reader_thread: Optional[threading.Thread] = field(
        default=None, init=False, repr=False
    )
    _loaded_model: Optional[str] = field(default=None, init=False, repr=False)
    _last_params: Optional[LoadParams] = field(
        default=None, init=False, repr=False
    )
    _exit_code: Optional[int] = field(default=None, init=False, repr=False)

    def _env(self, name: str, default: str = "") -> str:
        return os.environ.get(name, default).strip()

    def resolve_llama_server(self) -> str:
        """llama-server 実行ファイルを解決する。

        Returns
        -------
        str
            実行ファイルパス。

        Raises
        ------
        ProcessManagerError
            見つからない場合。
        """
        if self.llama_server:
            path = Path(self.llama_server)
            if path.is_file():
                return str(path)
            raise ProcessManagerError(
                f"llama-server が見つかりません: {self.llama_server}", status=500
            )

        env = self._env("LLAMACPP_CHAT2_LLAMA_SERVER")
        if env and Path(env).is_file():
            return env

        found = shutil.which("llama-server")
        if found:
            return found

        root = self._env("LLAMACPP_CHAT2_ROOT")
        if root:
            for rel in (
                "llama-server",
                "llama-server.exe",
                "build/bin/llama-server",
                "build/bin/llama-server.exe",
                "build/bin/Release/llama-server.exe",
            ):
                candidate = Path(root) / rel
                if candidate.is_file():
                    return str(candidate)

        raise ProcessManagerError(
            "llama-server が見つかりません。"
            " LLAMACPP_CHAT2_LLAMA_SERVER を設定するか PATH に追加してください。",
            status=500,
        )

    def resolve_model_path(self, model: str) -> Path:
        """モデル ID またはパスを実ファイルへ解決する。

        Parameters
        ----------
        model : str
            カタログ ID または GGUF パス。

        Returns
        -------
        pathlib.Path
            存在するモデルファイル。

        Raises
        ------
        ProcessManagerError
            解決できない、またはファイル欠落。
        """
        name = (model or "").strip()
        if not name:
            raise ProcessManagerError("model が空です")

        direct = Path(name)
        if direct.is_file():
            return direct.resolve()

        catalog = load_catalog(self.models_dir)
        for it in catalog["models"]:
            if it["id"] == name:
                path = self.models_dir / it["filename"]
                if not path.is_file():
                    raise ProcessManagerError(
                        f"モデルファイルが欠落しています: {path}", status=404
                    )
                return path

        # ファイル名のみ指定
        by_name = self.models_dir / Path(name).name
        if by_name.is_file():
            return by_name.resolve()

        raise ProcessManagerError(
            f"モデルが見つかりません: {name}", status=404
        )

    def resolve_mmproj_path(self, mmproj: Optional[str]) -> Optional[Path]:
        """mmproj パスを解決する。未指定なら None。

        Parameters
        ----------
        mmproj : str or None
            プロジェクター ID / パス / ファイル名。

        Returns
        -------
        pathlib.Path or None
            存在する mmproj ファイル。未指定時は None。

        Raises
        ------
        ProcessManagerError
            指定されたが見つからない場合。
        """
        value = (mmproj or "").strip()
        if not value:
            return None

        direct = Path(value)
        if direct.is_file():
            return direct.resolve()

        catalog = load_catalog(self.models_dir)
        for it in catalog["models"]:
            if it["id"] == value or it["filename"] == value:
                path = self.models_dir / it["filename"]
                if not path.is_file():
                    raise ProcessManagerError(
                        f"mmproj ファイルが欠落しています: {path}",
                        status=404,
                    )
                return path

        by_name = self.models_dir / Path(value).name
        if by_name.is_file():
            return by_name.resolve()

        raise ProcessManagerError(
            f"mmproj が見つかりません: {value}", status=404
        )

    def _append_log(self, text: str) -> None:
        if not text:
            return
        for line in text.splitlines():
            self._log_lines.append(line)
        while self._log_tail_char_count() > LOG_TAIL_MAX_CHARS:
            if self._log_lines:
                self._log_lines.popleft()

    def _log_tail_char_count(self) -> int:
        return sum(len(line) + 1 for line in self._log_lines)

    def log_tail(self) -> str:
        """ログ末尾を返す。"""
        with self._lock:
            return "\n".join(self._log_lines)

    def _popen_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        return kwargs

    def _start_reader(self, proc: subprocess.Popen[str]) -> None:
        def _reader() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    with self._lock:
                        self._append_log(line.rstrip("\n"))
            except (OSError, ValueError):
                pass
            finally:
                with self._lock:
                    code = proc.poll()
                    if code is not None:
                        self._exit_code = code
                        if self._proc is proc:
                            self._proc = None

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        self._reader_thread = thread

    def _build_command(
        self,
        model_path: Path,
        *,
        ngl: int,
        ctx: int,
        mmproj_path: Optional[Path],
        host: str,
        port: int,
    ) -> list[str]:
        exe = self.resolve_llama_server()
        cmd = [
            exe,
            "-m",
            str(model_path),
            "--host",
            host,
            "--port",
            str(port),
            "--n-gpu-layers",
            str(int(ngl)),
            "--ctx-size",
            str(int(ctx)),
        ]
        if mmproj_path is not None:
            cmd.extend(["--mmproj", str(mmproj_path)])
        return cmd

    def _is_running_locked(self) -> bool:
        if self._proc is None:
            return False
        code = self._proc.poll()
        if code is None:
            return True
        self._exit_code = code
        self._proc = None
        return False

    def is_running(self) -> bool:
        """子プロセスが稼働中か。"""
        with self._lock:
            return self._is_running_locked()

    def unload(self) -> dict[str, Any]:
        """子プロセスを停止する。未ロードなら no-op。

        Returns
        -------
        dict
            操作結果。
        """
        with self._lock:
            if not self._is_running_locked():
                return {
                    "ok": True,
                    "message": "未ロード（no-op）",
                    "llama_running": False,
                }

            proc = self._proc
            assert proc is not None
            proc.terminate()
            try:
                proc.wait(timeout=TERMINATE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)

            code = proc.poll()
            self._exit_code = code
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
            self._proc = None
            self._loaded_model = None
            return {
                "ok": True,
                "message": "unload 完了",
                "llama_running": False,
                "exit_code": code,
            }

    def load(
        self,
        model: str,
        *,
        ngl: int = 99,
        ctx: int = 8192,
        mmproj: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> dict[str, Any]:
        """モデルをロードする（既存プロセスは先に unload）。

        Parameters
        ----------
        model : str
            モデル ID またはパス。
        ngl : int, default 99
            GPU 層数。
        ctx : int, default 8192
            コンテキスト長。
        mmproj : str or None, default None
            任意 mmproj。未指定時は付けない。
        host : str or None
            バインドホスト。
        port : int or None
            推論ポート。

        Returns
        -------
        dict
            起動結果。

        Raises
        ------
        ProcessManagerError
            起動失敗。
        """
        bind_host = (host or self.llama_host or DEFAULT_LLAMA_HOST).strip()
        bind_port = int(port if port is not None else self.llama_port)

        with self._lock:
            if self._is_running_locked():
                self.unload()

            model_path = self.resolve_model_path(model)
            mmproj_path = self.resolve_mmproj_path(mmproj)
            cmd = self._build_command(
                model_path,
                ngl=ngl,
                ctx=ctx,
                mmproj_path=mmproj_path,
                host=bind_host,
                port=bind_port,
            )

            self._log_lines.clear()
            self._exit_code = None
            self._append_log(f"$ {' '.join(cmd)}")

            try:
                proc = subprocess.Popen(cmd, **self._popen_kwargs())
            except OSError as e:
                raise ProcessManagerError(
                    f"llama-server の起動に失敗: {e}", status=500
                ) from e

            self._proc = proc
            self._loaded_model = model
            self._last_params = LoadParams(
                model=model,
                ngl=int(ngl),
                ctx=int(ctx),
                mmproj=(mmproj or "").strip() or None,
                host=bind_host,
                port=bind_port,
            )
            self._start_reader(proc)

        time.sleep(STARTUP_WAIT_SEC)

        with self._lock:
            if not self._is_running_locked():
                tail = self.log_tail()
                raise ProcessManagerError(
                    "llama-server が起動直後に終了しました。"
                    f" exit_code={self._exit_code}\n{tail}",
                    status=500,
                )

            return {
                "ok": True,
                "message": "load 完了",
                "llama_running": True,
                "loaded_model": self._loaded_model,
                "model_path": str(model_path),
                "mmproj": str(mmproj_path) if mmproj_path else None,
                "host": bind_host,
                "port": bind_port,
                "pid": self._proc.pid if self._proc else None,
            }

    def restart(self) -> dict[str, Any]:
        """直近と同じ設定で再起動する。

        Returns
        -------
        dict
            操作結果。

        Raises
        ------
        ProcessManagerError
            直近 load 情報が無い場合。
        """
        with self._lock:
            params = self._last_params
        if params is None:
            raise ProcessManagerError(
                "再起動対象がありません（先に load してください）", status=400
            )
        self.unload()
        return self.load(
            params.model,
            ngl=params.ngl,
            ctx=params.ctx,
            mmproj=params.mmproj,
            host=params.host,
            port=params.port,
        )

    def start(self, **kwargs: Any) -> dict[str, Any]:
        """起動する。model があれば load、無ければ restart。

        Parameters
        ----------
        **kwargs
            load に渡す引数。

        Returns
        -------
        dict
            操作結果。
        """
        model = kwargs.pop("model", None)
        if model:
            return self.load(str(model), **kwargs)
        if self.is_running():
            return {
                "ok": True,
                "message": "既に稼働中",
                "llama_running": True,
                "loaded_model": self._loaded_model,
            }
        return self.restart()

    def stop(self) -> dict[str, Any]:
        """停止する（unload の別名）。"""
        return self.unload()

    def status_payload(self) -> dict[str, Any]:
        """status API 用の辞書を返す。"""
        with self._lock:
            running = self._is_running_locked()
            payload: dict[str, Any] = {
                "ok": True,
                "llama_running": running,
                "loaded_model": self._loaded_model if running else None,
                "exit_code": self._exit_code,
                "log_tail": self.log_tail(),
                "inference_host": self.llama_host,
                "inference_port": self.llama_port,
            }
            if not running and self._loaded_model and self._exit_code is not None:
                payload["last_loaded_model"] = self._loaded_model
                payload["message"] = (
                    f"llama-server は終了しています（exit_code={self._exit_code}）"
                )
            elif running:
                payload["message"] = "llama-server 稼働中"
            else:
                payload["message"] = "llama-server 未ロード"
            if self._last_params is not None:
                payload["last_load"] = {
                    "model": self._last_params.model,
                    "ngl": self._last_params.ngl,
                    "ctx": self._last_params.ctx,
                    "mmproj": self._last_params.mmproj,
                }
            return payload


def default_process_manager(models_dir: Path) -> ProcessManager:
    """環境変数から ProcessManager を組み立てる。

    Parameters
    ----------
    models_dir : pathlib.Path
        モデル一時ディレクトリ。

    Returns
    -------
    ProcessManager
        設定済みマネージャ。
    """
    host = os.environ.get(
        "LLAMACPP_CHAT2_LLAMA_HOST", DEFAULT_LLAMA_HOST
    ).strip() or DEFAULT_LLAMA_HOST
    port_raw = os.environ.get(
        "LLAMACPP_CHAT2_LLAMA_PORT", str(DEFAULT_LLAMA_PORT)
    ).strip()
    try:
        port = int(port_raw or str(DEFAULT_LLAMA_PORT))
    except ValueError:
        port = DEFAULT_LLAMA_PORT
    llama_server = os.environ.get("LLAMACPP_CHAT2_LLAMA_SERVER", "").strip() or None
    return ProcessManager(
        models_dir=models_dir,
        llama_server=llama_server,
        llama_host=host,
        llama_port=port,
    )
