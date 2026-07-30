"""サーバー側ユニットテスト（標準ライブラリ unittest）。"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.control_api import ControlState, make_handler  # noqa: E402
from server.models_catalog import (  # noqa: E402
    CatalogError,
    delete_model,
    download_model,
    list_models_with_status,
    normalize_repo_url,
    resolve_download_url,
    save_catalog,
    upsert_entry,
)
from server.process_manager import ProcessManager  # noqa: E402

FAKE_LLAMA = Path(__file__).resolve().parent / "test_support" / "fake_llama_server.py"
TOKEN = "test-token"


class ModelsCatalogTests(unittest.TestCase):
    """models_catalog のテスト。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_resolve_download_url_hf_repo(self) -> None:
        url = "https://huggingface.co/org/repo"
        got = resolve_download_url(url, "model-Q4.gguf")
        self.assertIn("/org/repo/resolve/main/model-Q4.gguf", got)

    def test_normalize_repo_url_org_repo(self) -> None:
        got = normalize_repo_url("huihui-ai/Some-GGUF")
        self.assertEqual("https://huggingface.co/huihui-ai/Some-GGUF", got)

    def test_normalize_repo_url_host_without_scheme(self) -> None:
        got = normalize_repo_url("huggingface.co/org/repo")
        self.assertEqual("https://huggingface.co/org/repo", got)

    def test_resolve_download_url_org_repo_short(self) -> None:
        got = resolve_download_url("org/repo", "model-Q4.gguf")
        self.assertIn("/org/repo/resolve/main/model-Q4.gguf", got)
        self.assertTrue(got.startswith("https://huggingface.co/"))

    def test_resolve_download_url_direct(self) -> None:
        url = "https://example.com/files/foo.gguf"
        self.assertEqual(url, resolve_download_url(url, "foo.gguf"))

    def test_list_models_missing_flag(self) -> None:
        upsert_entry(
            self.models_dir,
            model_id="m1",
            url="https://example.com/r",
            filename="a.gguf",
            overwrite=True,
        )
        items = list_models_with_status(self.models_dir)
        self.assertEqual(1, len(items))
        self.assertTrue(items[0]["missing"])

    def test_upsert_conflict(self) -> None:
        upsert_entry(
            self.models_dir,
            model_id="dup",
            url="https://example.com/r",
            filename="a.gguf",
            overwrite=True,
        )
        with self.assertRaises(CatalogError):
            upsert_entry(
                self.models_dir,
                model_id="dup",
                url="https://example.com/r2",
                filename="b.gguf",
            )

    def test_download_with_mmproj(self) -> None:
        from unittest import mock

        def fake_download(dest: Path, dl_url: str, *, timeout_sec: float) -> str:
            dest.write_bytes(b"GGUF")
            return dl_url

        with mock.patch(
            "server.models_catalog._download_file", side_effect=fake_download
        ):
            result = download_model(
                self.models_dir,
                model_id="Foo",
                url="org/repo",
                filename="Foo-Q4.gguf",
                mmproj_filename="mmproj-model.gguf",
            )
        self.assertTrue(result["ok"])
        self.assertEqual("Foo", result["model"]["id"])
        self.assertEqual(
            "https://huggingface.co/org/repo", result["model"]["url"]
        )
        self.assertEqual("Foo mmproj", result["mmproj"]["id"])
        self.assertTrue((self.models_dir / "Foo-Q4.gguf").is_file())
        self.assertTrue((self.models_dir / "mmproj-model.gguf").is_file())
        items = list_models_with_status(self.models_dir)
        self.assertEqual(2, len(items))

    def test_delete_model_removes_file(self) -> None:
        path = self.models_dir / "gone.gguf"
        path.write_bytes(b"GGUF")
        upsert_entry(
            self.models_dir,
            model_id="Gone",
            url="https://example.com/r",
            filename="gone.gguf",
            overwrite=True,
        )
        result = delete_model(self.models_dir, "Gone", delete_file=True)
        self.assertTrue(result["ok"])
        self.assertFalse(path.exists())
        self.assertEqual([], list_models_with_status(self.models_dir))

    def test_delete_model_catalog_only_when_missing(self) -> None:
        upsert_entry(
            self.models_dir,
            model_id="Missing",
            url="https://example.com/r",
            filename="no-file.gguf",
            overwrite=True,
        )
        result = delete_model(self.models_dir, "Missing", delete_file=True)
        self.assertTrue(result["ok"])
        self.assertIsNone(result["file_removed"])
        self.assertEqual([], list_models_with_status(self.models_dir))


class ProcessManagerTests(unittest.TestCase):
    """process_manager のテスト。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)
        self.model_file = self.models_dir / "demo.gguf"
        self.model_file.write_bytes(b"GGUF")
        save_catalog(
            self.models_dir,
            {
                "version": 1,
                "models": [
                    {
                        "id": "Demo",
                        "url": "https://example.com/r",
                        "filename": "demo.gguf",
                    }
                ],
            },
        )
        self.pm = ProcessManager(
            models_dir=self.models_dir,
            llama_server=sys.executable,
            llama_host="127.0.0.1",
            llama_port=18080,
        )

    def tearDown(self) -> None:
        self.pm.unload()

    def test_resolve_model_by_id_and_path(self) -> None:
        by_id = self.pm.resolve_model_path("Demo")
        self.assertEqual(self.model_file.resolve(), by_id)
        by_path = self.pm.resolve_model_path(str(self.model_file))
        self.assertEqual(self.model_file.resolve(), by_path)

    def test_load_without_mmproj(self) -> None:
        original = ProcessManager._build_command

        def _patched(
            self,
            model_path: Path,
            *,
            ngl: int,
            ctx: int,
            mmproj_path: Path | None,
            host: str,
            port: int,
        ) -> list[str]:
            assert mmproj_path is None
            return [sys.executable, str(FAKE_LLAMA), str(model_path)]

        ProcessManager._build_command = _patched  # type: ignore[method-assign]
        try:
            result = self.pm.load("Demo", ngl=12, ctx=4096)
            self.assertTrue(result["ok"])
            self.assertTrue(self.pm.is_running())
            self.assertIsNone(result["mmproj"])
            status = self.pm.status_payload()
            self.assertTrue(status["llama_running"])
            self.assertEqual("Demo", status["loaded_model"])
        finally:
            ProcessManager._build_command = original  # type: ignore[method-assign]

    def test_unload_noop_and_restart(self) -> None:
        noop = self.pm.unload()
        self.assertTrue(noop["ok"])
        original = ProcessManager._build_command

        def _patched(self, model_path, **kwargs):  # type: ignore[no-untyped-def]
            return [sys.executable, str(FAKE_LLAMA), str(model_path)]

        ProcessManager._build_command = _patched  # type: ignore[method-assign]
        try:
            self.pm.load("Demo", ngl=1, ctx=512)
            restarted = self.pm.restart()
            self.assertTrue(restarted["ok"])
            self.assertTrue(self.pm.is_running())
        finally:
            ProcessManager._build_command = original  # type: ignore[method-assign]


class ControlApiHttpTests(unittest.TestCase):
    """control_api の HTTP 統合テスト。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.tmp.name)
        (self.models_dir / "demo.gguf").write_bytes(b"GGUF")
        save_catalog(
            self.models_dir,
            {
                "version": 1,
                "models": [
                    {
                        "id": "Demo",
                        "url": "https://example.com/r",
                        "filename": "demo.gguf",
                    }
                ],
            },
        )
        self.pm = ProcessManager(
            models_dir=self.models_dir,
            llama_server=sys.executable,
            llama_port=18081,
        )
        self.state = ControlState(
            token=TOKEN,
            models_dir=self.models_dir,
            process_manager=self.pm,
        )
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.state)
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.pm.unload()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        token: str = TOKEN,
    ) -> tuple[int, dict]:
        data = None
        headers = {
            "Accept": "application/json",
            "X-Control-Token": token,
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_status_requires_token(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._request("GET", "/v1/control/status", token="bad")
        self.assertEqual(401, ctx.exception.code)

    def test_status_and_models(self) -> None:
        code, body = self._request("GET", "/v1/control/status")
        self.assertEqual(200, code)
        self.assertTrue(body["ok"])
        self.assertFalse(body["llama_running"])

        code, body = self._request("GET", "/v1/control/models")
        self.assertEqual(200, code)
        self.assertEqual(1, len(body["models"]))
        self.assertFalse(body["models"][0]["missing"])

    def test_load_unload_via_http(self) -> None:
        original = ProcessManager._build_command

        def _patched(self, model_path, **kwargs):  # type: ignore[no-untyped-def]
            return [sys.executable, str(FAKE_LLAMA), str(model_path)]

        ProcessManager._build_command = _patched  # type: ignore[method-assign]
        try:
            code, body = self._request(
                "POST",
                "/v1/control/load",
                {"model": "Demo", "ngl": 5, "ctx": 2048},
            )
            self.assertEqual(200, code)
            self.assertTrue(body["llama_running"])
            self.assertIsNone(body.get("mmproj"))

            code, body = self._request("POST", "/v1/control/unload", {})
            self.assertEqual(200, code)
            self.assertFalse(body["llama_running"])
        finally:
            ProcessManager._build_command = original  # type: ignore[method-assign]

    def test_download_with_mmproj_via_http(self) -> None:
        from unittest import mock

        def fake_download(dest: Path, dl_url: str, *, timeout_sec: float) -> str:
            dest.write_bytes(b"GGUF")
            return dl_url

        with mock.patch(
            "server.models_catalog._download_file", side_effect=fake_download
        ):
            code, body = self._request(
                "POST",
                "/v1/control/models/download",
                {
                    "id": "Vis",
                    "url": "org/repo",
                    "filename": "vis-Q4.gguf",
                    "mmproj_filename": "mmproj.gguf",
                },
            )
        self.assertEqual(200, code)
        self.assertTrue(body["ok"])
        self.assertEqual("Vis mmproj", body["mmproj"]["id"])
        self.assertIn("huggingface.co/org/repo", body["model"]["url"])

    def test_delete_via_http(self) -> None:
        model_path = self.models_dir / "demo.gguf"
        self.assertTrue(model_path.is_file())
        code, body = self._request(
            "POST",
            "/v1/control/models/delete",
            {"id": "Demo", "delete_file": True},
        )
        self.assertEqual(200, code)
        self.assertTrue(body["ok"])
        code, body = self._request("GET", "/v1/control/models")
        self.assertEqual(200, code)
        self.assertEqual([], body["models"])
        self.assertFalse(model_path.exists())


if __name__ == "__main__":
    unittest.main()
