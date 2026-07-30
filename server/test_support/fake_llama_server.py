"""テスト用の llama-server 代替スクリプト。

起動後に 1 行出力して待機する。SIGTERM / terminate で終了する。
"""

from __future__ import annotations

import signal
import sys
import time


def main() -> None:
    print("fake llama-server ready", flush=True)
    running = True

    def _stop(*_args: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _stop)

    while running:
        time.sleep(0.05)

    print("fake llama-server exiting", flush=True)


if __name__ == "__main__":
    main()
