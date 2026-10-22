from types import SimpleNamespace
import pytest
from loki.ai import llm


def _reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _stream(*texts):
    return iter(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))]) for t in texts)


@pytest.fixture
def default_connection(monkeypatch):
    """LOKI's bundled default: fallbacks enabled."""
    monkeypatch.setattr(llm, "is_ai_customized", lambda model: False)
    monkeypatch.setattr(llm, "resolve_ai_connection", lambda model: {"model": "gemini/gemini-flash-latest"})


@pytest.fixture
def custom_connection(monkeypatch):
    """User-configured provider: exactly one attempt, never a silent provider switch."""
    monkeypatch.setattr(llm, "is_ai_customized", lambda model: True)
    monkeypatch.setattr(llm, "resolve_ai_connection", lambda model: {"model": "ollama/llama3", "api_base": "http://localhost:11434"})


def _fake_completion(monkeypatch, outcomes):
    """Each call pops the next outcome: an Exception is raised, anything else is returned."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm.litellm, "completion", fake)
    return calls


def test_default_connection_tries_gemini_siblings(default_connection):
    models = [a["model"] for a in llm.connection_attempts()]
    assert models[0] == "gemini/gemini-flash-latest"
    assert models[1:] == list(llm.DEFAULT_FALLBACK_MODELS)


def test_custom_connection_never_falls_back(custom_connection, monkeypatch):
    calls = _fake_completion(monkeypatch, [RuntimeError("503 unavailable")])
    with pytest.raises(llm.LLMError):
        llm.complete("hi")
    assert len(calls) == 1
    assert calls[0]["api_base"] == "http://localhost:11434"


def test_complete_falls_back_then_succeeds(default_connection, monkeypatch):
    calls = _fake_completion(monkeypatch, [RuntimeError("boom"), _reply("diagnosis")])
    assert llm.complete("hi") == "diagnosis"
    assert [c["model"] for c in calls] == ["gemini/gemini-flash-latest", llm.DEFAULT_FALLBACK_MODELS[0]]
    assert calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_complete_error_carries_last_cause(default_connection, monkeypatch):
    errors = [RuntimeError(f"fail {i}") for i in range(4)]
    _fake_completion(monkeypatch, list(errors))
    with pytest.raises(llm.LLMError) as exc:
        llm.complete("hi")
    assert exc.value.last_error is errors[-1]


@pytest.mark.parametrize("text", [
    '[{"status": "PASSED"}]',
    '```json\n[{"status": "PASSED"}]\n```',
    '```\n[{"status": "PASSED"}]\n```',
])
def test_extract_json_tolerates_fences(text):
    assert llm.extract_json(text) == [{"status": "PASSED"}]


def test_extract_json_keeps_fences_inside_values():
    text = '{"original_snippet": "```js\\ncode\\n```"}'
    assert llm.extract_json(text)["original_snippet"] == "```js\ncode\n```"


def test_complete_json_retries_on_invalid_reply(default_connection, monkeypatch):
    calls = _fake_completion(monkeypatch, [_reply("not json"), _reply('{"a": 1}'), _reply('[1]')])
    assert llm.complete_json("hi", validate=lambda d: isinstance(d, list)) == [1]
    assert len(calls) == 3
    assert all(c["temperature"] == 0.1 for c in calls)


def test_open_stream_yields_all_chunks(default_connection, monkeypatch):
    _fake_completion(monkeypatch, [_stream("", "Hel", None, "lo")])
    assert "".join(llm.open_stream([{"role": "user", "content": "hi"}])) == "Hello"


def test_open_stream_skips_transient_errors(default_connection, monkeypatch):
    calls = _fake_completion(monkeypatch, [RuntimeError("503 Service Unavailable"), _stream("ok")])
    assert list(llm.open_stream([])) == ["ok"]
    assert len(calls) == 2


def test_open_stream_stops_on_fatal_error(default_connection, monkeypatch):
    calls = _fake_completion(monkeypatch, [RuntimeError("401 invalid api key")])
    with pytest.raises(llm.LLMError) as exc:
        llm.open_stream([])
    assert not isinstance(exc.value, llm.LLMUnavailableError)
    assert len(calls) == 1


def test_open_stream_all_transient_is_unavailable(default_connection, monkeypatch):
    _fake_completion(monkeypatch, [RuntimeError("timeout")] * 4)
    with pytest.raises(llm.LLMUnavailableError):
        llm.open_stream([])


@pytest.mark.parametrize("cause, expected", [
    (ValueError("Missing API key.\nConfigure a key first."), "Missing API key. Configure a key first."),
    (TimeoutError("Request timed out.\nTry again later."), "Request timed out. Try again later."),
    (ConnectionError("Could not connect to provider."), "Could not connect to provider."),
])
def test_error_summary_is_one_line_and_preserves_cause(cause, expected):
    error = llm.LLMError(str(cause), cause)
    assert error.summary == f"{type(cause).__name__}: {expected}"
    assert error.last_error is cause
    assert not error.is_rate_limited


def test_error_summary_extracts_provider_message_without_payload():
    class RateLimitError(Exception):
        status_code = 429

    cause = RateLimitError(
        'provider - {"error": {"code": 429, "message": "Free-tier quota exceeded.\\nRetry later.", '
        '"status": "RESOURCE_EXHAUSTED", "details": [{"debug": "synthetic-payload-canary"}]}}'
    )
    error = llm.LLMError(str(cause), cause)
    assert error.summary == "RateLimitError: Free-tier quota exceeded. Retry later."
    assert error.is_rate_limited
    assert error.last_error is cause
    assert "synthetic-payload-canary" not in error.summary
    assert "{" not in error.summary


@pytest.mark.parametrize("message", [
    'RESOURCE_EXHAUSTED: {"error": {"message": "Quota exhausted"}}',
    '{"error": {"code": 429, "message": "Too many requests"}}',
    'litellm.RateLimitError: request limit reached',
    'HTTP 429: Too many requests',
])
def test_rate_limit_detection_without_concrete_provider_exception(message):
    error = llm.LLMError(message, RuntimeError(message))
    assert error.is_rate_limited


def test_rate_limit_detection_uses_http_status():
    cause = RuntimeError("Too many requests")
    cause.status_code = 429
    assert llm.LLMError(str(cause), cause).is_rate_limited


@pytest.mark.parametrize("message", [
    'provider - {"error": broken-json',
    'provider - {"error": {"details": ["synthetic-payload-canary"]}}',
])
def test_error_summary_does_not_dump_unusable_json(message):
    error = llm.LLMError(message, RuntimeError(message))
    assert error.summary == "RuntimeError: provider -"
    assert "{" not in error.summary


def test_error_summary_is_bounded_and_supports_message_only_errors():
    error = llm.LLMError("x" * 1000)
    assert len(error.summary) <= 240
    assert error.summary.endswith("...")
    assert error.last_error is None
    assert not error.is_rate_limited


def test_deeply_nested_diagnostic_json_does_not_hide_original_error():
    raw = 'provider - ' + '{"error":' * 2000 + '0' + '}' * 2000
    cause = RuntimeError(raw)
    error = llm.LLMError(raw, cause)
    assert error.summary == "RuntimeError: provider -"
    assert error.last_error is cause


def test_unrelated_number_is_not_an_http_status():
    assert not llm.LLMError("Received 429 bytes").is_rate_limited
