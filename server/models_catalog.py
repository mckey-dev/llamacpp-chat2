"""モデルカタログ（models.json）の読み書きとダウンロード。

標準ライブラリのみ。カタログ各エントリは ``id`` / ``url`` / ``filename``。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse, urlunparse
from urllib.request import Request, urlopen

CATALOG_NAME = "models.json"
CATALOG_VERSION = 1


class CatalogError(Exception):
    """カタログ操作の失敗。"""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def default_models_dir() -> Path:
    """既定のモデル一時ディレクトリを返す。

    Returns
    -------
    pathlib.Path
        ``LLAMACPP_CHAT2_MODELS_DIR`` または OS 一時配下。
    """
    env = os.environ.get("LLAMACPP_CHAT2_MODELS_DIR", "").strip()
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "llamacpp-chat2"


def catalog_path(models_dir: Path) -> Path:
    """カタログ JSON のパスを返す。"""
    return models_dir / CATALOG_NAME


def ensure_models_dir(models_dir: Path) -> Path:
    """モデルディレクトリを作成して返す。"""
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _empty_catalog() -> dict[str, Any]:
    return {"version": CATALOG_VERSION, "models": []}


def load_catalog(models_dir: Path) -> dict[str, Any]:
    """models.json を読み込む。無ければ空カタログ。

    Parameters
    ----------
    models_dir : pathlib.Path
        モデル一時ディレクトリ。

    Returns
    -------
    dict
        ``version`` と ``models`` を持つ辞書。
    """
    path = catalog_path(models_dir)
    if not path.is_file():
        return _empty_catalog()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CatalogError(f"models.json の読み込みに失敗: {e}", status=500) from e
    if not isinstance(raw, dict):
        return _empty_catalog()
    models = raw.get("models")
    if not isinstance(models, list):
        models = []
    cleaned: list[dict[str, str]] = []
    for it in models:
        if not isinstance(it, dict):
            continue
        mid = str(it.get("id") or "").strip()
        url = str(it.get("url") or "").strip()
        filename = str(it.get("filename") or "").strip()
        if mid and url and filename:
            cleaned.append({"id": mid, "url": url, "filename": filename})
    return {"version": int(raw.get("version") or CATALOG_VERSION), "models": cleaned}


def save_catalog(models_dir: Path, catalog: dict[str, Any]) -> Path:
    """models.json を保存する。

    Parameters
    ----------
    models_dir : pathlib.Path
        モデル一時ディレクトリ。
    catalog : dict
        カタログ全体。

    Returns
    -------
    pathlib.Path
        書き込んだパス。
    """
    ensure_models_dir(models_dir)
    path = catalog_path(models_dir)
    payload = {
        "version": int(catalog.get("version") or CATALOG_VERSION),
        "models": catalog.get("models") or [],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def sanitize_filename(filename: str) -> str:
    """ファイル名を検証し、ベース名のみ返す。

    Raises
    ------
    CatalogError
        不正なファイル名。
    """
    name = (filename or "").strip().replace("\\", "/")
    if not name or name in (".", "..") or "/" in name:
        raise CatalogError("filename が不正です（ディレクトリ区切りは不可）")
    base = Path(name).name
    if base != name or base in (".", ".."):
        raise CatalogError("filename が不正です")
    return base


def resolve_download_url(url: str, filename: str) -> str:
    """リポジトリ URL とファイル名から実ダウンロード URL を決める。

    - 末尾がファイル名、または ``.gguf`` / ``.bin`` 等で終わる直接 URL → そのまま
    - Hugging Face の ``/org/repo`` または ``/org/repo/tree/...`` →
      ``.../resolve/main/{filename}``

    Parameters
    ----------
    url : str
        リポジトリまたはファイル URL。
    filename : str
        保存ファイル名。

    Returns
    -------
    str
        ダウンロード URL。
    """
    u = (url or "").strip()
    if not u:
        raise CatalogError("url が空です")
    parsed = urlparse(u)
    if parsed.scheme not in ("http", "https"):
        raise CatalogError("url は http または https である必要があります")

    path = parsed.path or ""
    if path.rstrip("/").endswith("/" + filename) or path.endswith(filename):
        return u
    lower = path.lower()
    if lower.endswith((".gguf", ".bin", ".safetensors", ".ggml")):
        return u

    # Hugging Face リポジトリページ
    if parsed.netloc.endswith("huggingface.co"):
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            org, repo = parts[0], parts[1]
            # tree/blob 以降は捨てて resolve/main に寄せる
            file_enc = quote(filename)
            new_path = f"/{org}/{repo}/resolve/main/{file_enc}"
            return urlunparse(
                (parsed.scheme, parsed.netloc, new_path, "", parsed.query, "")
            )

    # それ以外: URL をディレクトリとみなし filename を連結
    base = u.rstrip("/")
    return f"{base}/{quote(filename)}"


def list_models_with_status(models_dir: Path) -> list[dict[str, Any]]:
    """カタログを読み、欠落フラグ付きで返す。

    Returns
    -------
    list of dict
        ``id`` / ``url`` / ``filename`` / ``missing`` / ``path``。
    """
    ensure_models_dir(models_dir)
    catalog = load_catalog(models_dir)
    out: list[dict[str, Any]] = []
    for it in catalog["models"]:
        fname = it["filename"]
        fpath = models_dir / fname
        missing = not fpath.is_file()
        out.append(
            {
                "id": it["id"],
                "url": it["url"],
                "filename": fname,
                "path": str(fpath),
                "missing": missing,
            }
        )
    return out


def upsert_entry(
    models_dir: Path,
    *,
    model_id: str,
    url: str,
    filename: str,
    overwrite: bool = False,
) -> dict[str, str]:
    """カタログにエントリを追加または上書きする（ファイル操作なし）。"""
    mid = (model_id or "").strip()
    u = (url or "").strip()
    fname = sanitize_filename(filename)
    if not mid:
        raise CatalogError("id が空です")
    if not u:
        raise CatalogError("url が空です")

    catalog = load_catalog(models_dir)
    models: list[dict[str, str]] = list(catalog["models"])
    idx = next((i for i, m in enumerate(models) if m["id"] == mid), None)
    entry = {"id": mid, "url": u, "filename": fname}
    if idx is None:
        models.append(entry)
    else:
        if not overwrite:
            raise CatalogError(
                f"id が既に存在します: {mid}（overwrite=true で上書き可）",
                status=409,
            )
        models[idx] = entry
    catalog["models"] = models
    save_catalog(models_dir, catalog)
    return entry


def download_model(
    models_dir: Path,
    *,
    model_id: str,
    url: str,
    filename: str,
    overwrite: bool = False,
    timeout_sec: float = 3600.0,
) -> dict[str, Any]:
    """モデルをダウンロードしてカタログに登録する。

    Parameters
    ----------
    models_dir : pathlib.Path
        保存先ディレクトリ。
    model_id : str
        タイトル（ID）。
    url : str
        リポジトリまたはファイル URL。
    filename : str
        保存ファイル名。
    overwrite : bool, default False
        同一 id の上書きを許すか。
    timeout_sec : float, default 3600.0
        ダウンロード全体のタイムアウト秒。

    Returns
    -------
    dict
        登録エントリと保存パス。
    """
    ensure_models_dir(models_dir)
    fname = sanitize_filename(filename)
    mid = (model_id or "").strip()
    u = (url or "").strip()
    if not mid:
        raise CatalogError("id が空です")
    if not u:
        raise CatalogError("url が空です")

    catalog = load_catalog(models_dir)
    exists = any(m["id"] == mid for m in catalog["models"])
    if exists and not overwrite:
        raise CatalogError(
            f"id が既に存在します: {mid}（overwrite=true で上書き可）",
            status=409,
        )

    dest = models_dir / fname
    if dest.is_file() and not overwrite:
        # 別 id で同名ファイルがある場合も拒否
        raise CatalogError(
            f"ファイルが既に存在します: {fname}（overwrite=true で上書き可）",
            status=409,
        )

    dl_url = resolve_download_url(u, fname)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        req = Request(
            dl_url,
            headers={"User-Agent": "llamacpp-chat2-control/0.1"},
        )
        with urlopen(req, timeout=timeout_sec) as resp:
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        tmp.replace(dest)
    except HTTPError as e:
        _remove_quiet(tmp)
        raise CatalogError(
            f"ダウンロード HTTP {e.code}: {dl_url}", status=502
        ) from e
    except URLError as e:
        _remove_quiet(tmp)
        raise CatalogError(f"ダウンロード失敗: {e.reason}", status=502) from e
    except OSError as e:
        _remove_quiet(tmp)
        raise CatalogError(f"保存失敗: {e}", status=500) from e
    finally:
        if tmp.exists() and not dest.exists():
            _remove_quiet(tmp)

    entry = upsert_entry(
        models_dir,
        model_id=mid,
        url=u,
        filename=fname,
        overwrite=True,
    )
    return {
        "ok": True,
        "model": entry,
        "path": str(dest),
        "download_url": dl_url,
    }


def _remove_quiet(path: Path) -> None:
    """存在すれば削除し、失敗は無視する。"""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
