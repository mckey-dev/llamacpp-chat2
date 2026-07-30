"""接続設定の JSON 読み書き（パスは呼び出し側が注入）。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from core.types import ConnectionConfig

PathLike = Union[str, Path]


def default_config_path(root: Optional[PathLike] = None) -> Path:
    """既定の設定ファイルパスを返す。

    Parameters
    ----------
    root : path-like or None, default None
        リポジトリルート。省略時はカレントディレクトリ。

    Returns
    -------
    pathlib.Path
        ``frontend-config.json`` のパス。
    """
    base = Path(root) if root is not None else Path.cwd()
    return base / "frontend-config.json"


def _env_image_max_long_edge() -> Optional[int]:
    """環境変数から画像長辺上限を読む。未設定なら None。"""
    raw = os.environ.get("LLAMACPP_CHAT2_IMAGE_MAX_LONG_EDGE", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_image_max_long_edge(raw: Optional[dict[str, Any]] = None) -> int:
    """画像長辺上限を解決する。

    設定 JSON にキーがあればそれを使い、無ければ環境変数、
    それも無ければ 1024。

    Parameters
    ----------
    raw : dict or None
        JSON 由来の辞書。

    Returns
    -------
    int
        長い辺の上限ピクセル。
    """
    data = raw or {}
    if "image_max_long_edge" in data and data["image_max_long_edge"] is not None:
        return int(data["image_max_long_edge"])
    env_val = _env_image_max_long_edge()
    if env_val is not None:
        return env_val
    return int(ConnectionConfig.image_max_long_edge)


def load_config(path: Optional[PathLike] = None) -> ConnectionConfig:
    """接続設定を JSON から読み込む。

    Parameters
    ----------
    path : path-like or None, default None
        設定ファイル。省略時は :func:`default_config_path`。

    Returns
    -------
    ConnectionConfig
        読み込んだ設定。ファイルが無い場合は既定値。
    """
    p = Path(path) if path is not None else default_config_path()
    if not p.is_file():
        return ConnectionConfig(image_max_long_edge=resolve_image_max_long_edge())
    raw = json.loads(p.read_text(encoding="utf-8"))
    return config_from_dict(raw)


def save_config(cfg: ConnectionConfig, path: Optional[PathLike] = None) -> Path:
    """接続設定を JSON に保存する。

    Parameters
    ----------
    cfg : ConnectionConfig
        保存する設定。
    path : path-like or None, default None
        出力パス。省略時は既定パス。

    Returns
    -------
    pathlib.Path
        書き込んだパス。
    """
    p = Path(path) if path is not None else default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(config_to_dict(cfg), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return p


def config_to_dict(cfg: ConnectionConfig) -> dict[str, Any]:
    """設定を JSON 用 dict に変換する。

    Parameters
    ----------
    cfg : ConnectionConfig
        設定。

    Returns
    -------
    dict
        シリアライズ可能な辞書。
    """
    return {
        "inference_base_url": cfg.inference_base_url,
        "control_base_url": cfg.control_base_url,
        "control_token": cfg.control_token,
        "ngl": cfg.ngl,
        "ctx": cfg.ctx,
        "timeout_sec": cfg.timeout_sec,
        "stream_timeout_sec": cfg.stream_timeout_sec,
        "image_max_long_edge": cfg.image_max_long_edge,
    }


def config_from_dict(raw: dict[str, Any]) -> ConnectionConfig:
    """dict から設定オブジェクトを構築する。

    Parameters
    ----------
    raw : dict
        JSON 由来の辞書。

    Returns
    -------
    ConnectionConfig
        構築結果。
    """
    return ConnectionConfig(
        inference_base_url=str(
            raw.get("inference_base_url") or ConnectionConfig.inference_base_url
        ),
        control_base_url=str(
            raw.get("control_base_url") or ConnectionConfig.control_base_url
        ),
        control_token=str(
            raw.get("control_token") or ConnectionConfig.control_token
        ),
        ngl=int(raw.get("ngl", 99)),
        ctx=int(raw.get("ctx", 8192)),
        timeout_sec=float(raw.get("timeout_sec", 60.0)),
        stream_timeout_sec=float(raw.get("stream_timeout_sec", 300.0)),
        image_max_long_edge=resolve_image_max_long_edge(raw),
    )
