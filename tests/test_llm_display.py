"""LLM errors use the same concise formatting at each user-facing boundary."""
import io
import json

import pytest
from rich.console import Console

from loki.ai import chat, llm
from loki.ai.brain import AIBrain
from loki.models import IncidentReport


def _quota_error():
    return RuntimeError(
        'provider - {"error": {"code": 429, "message": "Quota exceeded", '
        '"details": ["synthetic-payload-canary"]}}'
    )


@pytest.mark.parametrize("quota", [True, False])
def test_diagnosis_falls_back_with_appropriate_tip(tmp_path, monkeypatch, quota):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    run = tmp_path / "runs" / "run_synthetic"
    run.mkdir(parents=True)
    (run / "incident.json").write_text(json.dumps({"target_url": "http://localhost:8000"}), encoding="utf-8")
    cause = _quota_error() if quota else ValueError("Missing API key")

    def fail(*args, **kwargs):
        raise llm.LLMError(str(cause), cause)

    monkeypatch.setattr(llm, "complete", fail)
    result = AIBrain(str(run.parent)).diagnose_and_fix("run_synthetic")
    text = result["diagnosis"]
    assert "Local Diagnosis (evidence only, no AI reasoning)" in text
    assert "synthetic-payload-canary" not in text
    if quota:
        assert "RuntimeError: Quota exceeded" in text
        assert "Wait for the quota/rate limit to reset" in text
        assert "Set `GEMINI_API_KEY`" not in text
    else:
        assert "ValueError: Missing API key" in text
        assert "Set `GEMINI_API_KEY`" in text


def test_business_rules_error_uses_summary(monkeypatch):
    cause = _quota_error()

    def fail(*args, **kwargs):
        raise llm.LLMError(str(cause), cause)

    monkeypatch.setattr(llm, "complete_json", fail)
    rows = AIBrain().evaluate_business_rules(IncidentReport("http://localhost:8000"), "Synthetic rule")
    assert rows[0]["status"] == "ERROR"
    assert rows[0]["observation"] == "AI model evaluation error: RuntimeError: Quota exceeded"


@pytest.mark.parametrize("mode", ["wrapped", "midstream", "timeout"])
def test_chat_formats_errors_and_discards_failed_turn(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    prompts = iter(["synthetic prompt", "exit"])
    monkeypatch.setattr(chat.LokiChatSession, "_read_input", lambda self: next(prompts))
    monkeypatch.setattr(chat.LokiChatSession, "_load_project_context", lambda self: "")
    monkeypatch.setattr(chat, "check_for_updates", lambda: None)
    cause = _quota_error()

    def partial():
        yield "partial response"
        raise cause

    def fail(*args, **kwargs):
        if mode == "midstream":
            return partial()
        if mode == "timeout":
            timeout = TimeoutError("Request timed out")
            raise llm.LLMUnavailableError(str(timeout), timeout)
        raise llm.LLMError(str(cause), cause)

    monkeypatch.setattr(llm, "open_stream", fail)
    output = io.StringIO()
    session = chat.LokiChatSession()
    session.console = Console(file=output, width=200, color_system=None)
    session.start()
    text = output.getvalue()
    assert "synthetic-payload-canary" not in text
    assert "TimeoutError: Request timed out" in text if mode == "timeout" else "RuntimeError: Quota exceeded" in text
    assert all(turn["role"] == "system" for turn in session.history)
