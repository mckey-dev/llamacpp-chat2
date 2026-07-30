"""HttpBackend ストリーム timings / tok_s のユニットテスト。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backend.http import HttpBackend, _tok_s_from_obj  # noqa: E402
from core.types import ChatMessage  # noqa: E402


class FakeResponse:
    """httpx stream 応答の簡易モック。"""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._lines = lines
        self.reason_phrase = "OK"

    def read(self) -> bytes:
        return b""

    def iter_lines(self) -> Iterator[str]:
        yield from self._lines

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeClient:
    def __init__(self, resp: FakeResponse) -> None:
        self._resp = resp

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def stream(self, *args: Any, **kwargs: Any) -> FakeResponse:
        return self._resp


class TokSStreamTests(unittest.TestCase):
    """timings → tok_s の抽出。"""

    def test_tok_s_from_obj(self) -> None:
        self.assertAlmostEqual(
            42.5,
            _tok_s_from_obj(
                {"timings": {"predicted_per_second": 42.5}}
            )
            or 0.0,
        )
        self.assertIsNone(_tok_s_from_obj({}))

    def test_chat_stream_reads_timings_chunk(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}',
            (
                'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],'
                '"timings":{"predicted_per_second":35.4}}'
            ),
            (
                'data: {"choices":[],"usage":{"completion_tokens":2},'
                '"timings":{"predicted_per_second":35.4}}'
            ),
            "data: [DONE]",
        ]
        resp = FakeResponse(lines)
        backend = HttpBackend("http://127.0.0.1:8080")
        chunks = []
        with mock.patch("httpx.Client", return_value=FakeClient(resp)):
            for c in backend.chat_stream(
                [ChatMessage(role="user", content="hi")]
            ):
                chunks.append(c)
        texts = "".join(c.text for c in chunks)
        self.assertEqual("Hi", texts)
        done = [c for c in chunks if c.done]
        self.assertTrue(done)
        self.assertAlmostEqual(35.4, done[-1].tok_s or 0.0)


if __name__ == "__main__":
    unittest.main()
