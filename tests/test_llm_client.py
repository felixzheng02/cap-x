"""Unit tests for the local-VLM path in capx/llm/client.py.

No GPU, no vLLM: a stub HTTP server stands in for the OpenAI-compatible endpoint.

    cd cap-x && .venv-client/bin/python -m pytest tests/test_llm_client.py -v
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from capx.llm.client import (
    DEFAULT_HTTP_TIMEOUT,
    VLM_MODELS,
    EmptyContentError,
    query_model,
    thinking_kwargs,
)
from capx.utils.launch_utils import _extract_code


@dataclass
class Args:
    model: str = "qwen3.8-27b"
    server_url: str = ""
    api_key: str | None = "EMPTY"
    temperature: float = 0.7
    max_tokens: int = 256
    reasoning_effort: str = "medium"
    debug: bool = False
    http_timeout: int = 30


class _Stub(BaseHTTPRequestHandler):
    status = 200
    body: dict = {}
    seen: list = []
    hits = 0

    def do_POST(self):
        _Stub.hits += 1
        n = int(self.headers.get("Content-Length", 0))
        _Stub.seen.append(json.loads(self.rfile.read(n)))
        payload = json.dumps(_Stub.body).encode()
        self.send_response(_Stub.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture
def stub():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Stub.seen, _Stub.hits, _Stub.status = [], 0, 200
    yield f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"
    srv.shutdown()


def _ok(content, reasoning=None, finish="stop"):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": finish}],
            "usage": {"completion_tokens": 123}}


# ------------------------------------------------------------------ registry

def test_local_model_is_registered_as_a_vlm():
    """If absent, trial.py silently drops images and asserts on image differencing."""
    assert "qwen3.8-27b" in VLM_MODELS


# ------------------------------------------------------- thinking controls

@pytest.mark.parametrize(
    "effort,expected",
    [
        ("off", {"enable_thinking": False}),
        ("none", {"enable_thinking": False}),
        ("minimal", {"enable_thinking": False}),
        ("low", {"reasoning_effort": "low"}),
        ("medium", {"reasoning_effort": "medium"}),
        ("xhigh", {"reasoning_effort": "xhigh"}),
        ("high", {"reasoning_effort": "xhigh"}),   # LaunchArgs documents 'high'
        (None, {"reasoning_effort": "medium"}),    # default
    ],
)
def test_thinking_kwargs_mapping(effort, expected):
    kwargs, _ = thinking_kwargs(effort)
    assert kwargs == expected


def test_payload_carries_chat_template_kwargs(stub):
    _Stub.body = _ok("print('hi')")
    query_model(Args(server_url=stub, reasoning_effort="low"), [{"role": "user", "content": "x"}])
    body = _Stub.seen[-1]
    assert body["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert body["top_p"] == 0.95 and body["top_k"] == 20      # thinking sampling
    assert body["max_tokens"] == 256 and body["model"] == "qwen3.8-27b"


def test_payload_switches_sampling_when_thinking_off(stub):
    _Stub.body = _ok("print('hi')")
    query_model(Args(server_url=stub, reasoning_effort="off"), [{"role": "user", "content": "x"}])
    body = _Stub.seen[-1]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["top_p"] == 0.8 and body["presence_penalty"] == 1.5


# ------------------------------------------------------------ empty content

def test_null_content_with_reasoning_raises(stub):
    """The critical one: None would otherwise become a silent 'finish' in multi-turn."""
    _Stub.body = _ok(None, reasoning="thinking...", finish="length")
    with pytest.raises(EmptyContentError, match="reasoning trace consumed"):
        query_model(Args(server_url=stub), [{"role": "user", "content": "x"}])


def test_empty_string_content_also_raises(stub):
    _Stub.body = _ok("", reasoning=None)
    with pytest.raises(EmptyContentError):
        query_model(Args(server_url=stub), [{"role": "user", "content": "x"}])


def test_reasoning_content_is_read(stub):
    _Stub.body = _ok("code", reasoning="because")
    out = query_model(Args(server_url=stub), [{"role": "user", "content": "x"}])
    assert out["content"] == "code" and out["reasoning"] == "because"


def test_legacy_reasoning_field_still_read(stub):
    _Stub.body = {"choices": [{"message": {"content": "c", "reasoning": "r"},
                               "finish_reason": "stop"}]}
    out = query_model(Args(server_url=stub), [{"role": "user", "content": "x"}])
    assert out["reasoning"] == "r"


# --------------------------------------------------------------- transport

def test_404_is_not_retried_and_names_the_fix(stub):
    """A wrong path used to sleep 240s per attempt, forever."""
    _Stub.status = 404
    _Stub.body = {"detail": "Not Found"}
    with pytest.raises(RuntimeError, match=r"/v1/chat/completions"):
        query_model(Args(server_url=stub), [{"role": "user", "content": "x"}])
    assert _Stub.hits == 1, "404 must not be retried"


def test_default_http_timeout_covers_a_full_generation():
    """20 tok/s x 4096 tokens ~= 205s; the historical 200s was below one full reply."""
    assert DEFAULT_HTTP_TIMEOUT >= 600


# ----------------------------------------------------------- code extraction

def test_extract_code_handles_none():
    assert _extract_code(None) == [""]


def test_extract_code_plain():
    assert _extract_code("x = 1\ny = 2") == ["x = 1\ny = 2"]


def test_extract_code_python_fence():
    assert _extract_code("blah\n```python\nx = 1\n```\ntrailing") == ["x = 1"]


def test_extract_code_bare_fence_drops_preamble():
    """A bare ``` fence previously swallowed the preamble, guaranteeing a SyntaxError."""
    assert _extract_code("Here you go:\n```\nx = 1\n```") == ["x = 1"]


# ------------------------------------------------------------- truncation

def test_console_truncation_keeps_head_and_tail():
    from capx.envs.trial import CONSOLE_TRUNCATE_CHARS, _truncate_console

    text = "A" * 50_000 + "TRACEBACK_TAIL"
    out = _truncate_console(text)
    assert len(out) < CONSOLE_TRUNCATE_CHARS + 200
    assert out.startswith("AAA") and out.endswith("TRACEBACK_TAIL")
    assert "characters omitted" in out


def test_console_truncation_passes_short_text_through():
    from capx.envs.trial import _truncate_console

    assert _truncate_console("short") == "short"
    assert _truncate_console("") == ""
    assert _truncate_console(None) == ""
