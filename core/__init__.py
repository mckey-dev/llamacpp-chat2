"""フロントエンド用コア（Gradio 非依存）。"""

from core.config import load_config, save_config
from core.types import ChatMessage, ConnectionConfig

__all__ = [
    "ChatMessage",
    "ConnectionConfig",
    "load_config",
    "save_config",
]
