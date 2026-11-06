import httpx
import pytest

from dronedocs import llm as llm_mod
from dronedocs.llm import ChatClient, LLMError, parse_json_loose

OK = {
    "model": "m",
    "choices": [{"message": {"content": " hi "}}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
}


def _client(handler, **kw):
    return ChatClient(
        model="m", base_url="http://test/v1", api_key="k", max_retries=3, transport=httpx.MockTransport(handler), **kw
    )


def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "rate"})
        return httpx.Response(200, json=OK)

    r = _client(handler).chat([{"role": "user", "content": "x"}])
    assert r.text == "hi" and r.retries == 2 and r.total_tokens == 10


def test_auth_error_is_not_retried():
    with pytest.raises(LLMError):
        _client(lambda req: httpx.Response(401, json={"error": "bad key"})).chat([{"role": "user", "content": "x"}])


def test_reasoning_effort_only_for_gpt_oss():
    c = ChatClient(model="openai/gpt-oss-120b", reasoning_effort="low")
    assert c._payload([], None, None, False)["reasoning_effort"] == "low"
    c2 = ChatClient(model="llama3.2", reasoning_effort="low")
    assert "reasoning_effort" not in c2._payload([], None, None, False)


def test_cache_roundtrip(tmp_path):
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        return httpx.Response(200, json=OK)

    c = _client(handler, cache_path=tmp_path / "c.sqlite")
    c.chat([{"role": "user", "content": "x"}])
    second = c.chat([{"role": "user", "content": "x"}])
    assert n["calls"] == 1 and second.cached


def test_parse_json_loose():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Sure: {"score": 0.5} done') == {"score": 0.5}
