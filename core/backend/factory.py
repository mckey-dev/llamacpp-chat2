"""ConnectionConfig からクライアントを生成する。"""

from __future__ import annotations

from core.backend.control_client import ControlClient
from core.backend.http import HttpBackend
from core.types import ConnectionConfig


def make_http_backend(cfg: ConnectionConfig) -> HttpBackend:
    """推論クライアントを生成する。

    Parameters
    ----------
    cfg : ConnectionConfig
        接続設定。

    Returns
    -------
    HttpBackend
        推論クライアント。
    """
    return HttpBackend(
        cfg.inference_base_url,
        timeout_sec=cfg.timeout_sec,
        stream_timeout_sec=cfg.stream_timeout_sec,
    )


def make_control_client(cfg: ConnectionConfig) -> ControlClient:
    """制御クライアントを生成する。

    Parameters
    ----------
    cfg : ConnectionConfig
        接続設定。

    Returns
    -------
    ControlClient
        制御クライアント。
    """
    return ControlClient(
        cfg.control_base_url,
        token=cfg.control_token,
        timeout_sec=cfg.timeout_sec,
    )
