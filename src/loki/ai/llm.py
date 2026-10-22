"""Single gateway to the LLM for every LOKI feature (chat, diagnosis, rules, healing).

Any LiteLLM-compatible provider works (Gemini, OpenAI, Anthropic, Mistral, Groq, Bedrock,
a local Ollama/vLLM server, any OpenAI-compatible endpoint), resolved by
`config.resolve_ai_connection()`. Sibling-model fallbacks, for transient "high demand"
errors, only apply to LOKI's own bundled Gemini default: once the user configured `ai:`
or --model, exactly that connection is tried and nothing else, so LOKI never silently
switches to a provider they didn't ask for.
"""
import json
import re
from collections.abc import Callable, Iterator
from typing import Any
import litellm
from loki.config import is_ai_customized, resolve_ai_connection

litellm.suppress_debug_info = True

DEFAULT_FALLBACK_MODELS = (
    "gemini/gemini-flash-lite-latest",
    "gemini/gemini-3.6-flash",
    "gemini/gemini-3.5-flash-lite",
)
DEFAULT_TIMEOUT = 25
_TRANSIENT_MARKERS = ("503", "unavailable", "timeout")
_JSON_FENCE = re.compile(r"^```[\w-]*[ \t]*\n(.*?)\n?```$", re.DOTALL)

Messages = list[dict[str, Any]]


def _error_summary(message: str, cause: Exception | None) -> tuple[str, bool]:
    """Keep the provider's main message, not its diagnostic JSON envelope."""
    raw = str(cause) if cause is not None else message
    kind = type(cause).__name__ if cause is not None else "LLMError"
    rate_limited = str(getattr(cause, "status_code", "")) == "429" or any(
        marker in f"{kind} {raw}".lower() for marker in ("ratelimiterror", "resource_exhausted")
    )
    rate_limited = rate_limited or bool(re.search(r"\b(?:HTTP(?: status)?|status(?:_code)?|error code)\s*[:=]?\s*429\b", raw, re.IGNORECASE))
    start = raw.find("{")
    if start >= 0:
        prefix = raw[:start].strip()
        try:
            payload, _ = json.JSONDecoder().raw_decode(raw[start:])
        except (json.JSONDecodeError, RecursionError):
            raw = prefix
        else:
            detail = payload.get("error", payload) if isinstance(payload, dict) else {}
            if isinstance(detail, dict):
                rate_limited = rate_limited or str(detail.get("code", "")) == "429"
                main_message = detail.get("message")
                raw = main_message if isinstance(main_message, str) else prefix
            else:
                raw = prefix
    concise = " ".join(raw.split())
    summary = f"{kind}: {concise}" if concise else kind
    return (summary if len(summary) <= 240 else summary[:237] + "..."), rate_limited


class LLMError(Exception):
    """Failed call with a display summary; `last_error` retains the underlying cause."""

    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error
        self.summary, self.is_rate_limited = _error_summary(message, last_error)


class LLMUnavailableError(LLMError):
    """Every attempt failed with a transient provider error (overload, outage, timeout)."""


def connection_attempts(model: str | None = None) -> list[dict[str, Any]]:
    """LiteLLM kwargs to try in order: the resolved connection, plus Gemini siblings if it is the default."""
    base = resolve_ai_connection(model)
    if is_ai_customized(model):
        return [base]
    return [base] + [{**base, "model": m} for m in DEFAULT_FALLBACK_MODELS if m != base.get("model")]


def is_transient(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def extract_json(text: str | None) -> Any:
    """Parses a JSON reply, tolerating a single surrounding ``` / ```json markdown fence."""
    raw = (text or "").strip()
    fenced = _JSON_FENCE.match(raw)
    if fenced:
        raw = fenced.group(1).strip()
    return json.loads(raw)


def _as_messages(prompt: str | Messages) -> Messages:
    return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt


def complete(
    prompt: str | Messages,
    model: str | None = None,
    parse: Callable[[str], Any] | None = None,
    **options: Any,
) -> Any:
    """Returns the first successful completion (passed through `parse` if given).

    A `parse` exception counts as a failed attempt, so a malformed reply moves on to the
    next fallback model. Raises LLMError once every attempt has failed.
    """
    options = {"timeout": DEFAULT_TIMEOUT, "num_retries": 1, **options}
    last_error: Exception | None = None
    for kwargs in connection_attempts(model):
        try:
            response = litellm.completion(messages=_as_messages(prompt), **options, **kwargs)
            text = response.choices[0].message.content or ""
            return parse(text) if parse else text
        except Exception as e:
            last_error = e
    raise LLMError(str(last_error), last_error)


def complete_json(
    prompt: str | Messages,
    model: str | None = None,
    validate: Callable[[Any], bool] | None = None,
    **options: Any,
) -> Any:
    """Like complete(), but returns parsed JSON; replies failing `validate` count as failed attempts."""

    def parse(text: str) -> Any:
        data = extract_json(text)
        if validate and not validate(data):
            raise ValueError("LLM reply did not match the expected JSON schema")
        return data

    return complete(prompt, model=model, parse=parse, temperature=options.pop("temperature", 0.1), **options)


def open_stream(messages: Messages, model: str | None = None, **options: Any) -> Iterator[str]:
    """Starts a streamed completion and blocks until its first text chunk arrives.

    Transient provider errors fall through to the next fallback model; any other error
    is raised right away as LLMError. Returns an iterator over all text chunks, first included.
    """
    options = {"timeout": DEFAULT_TIMEOUT, "num_retries": 1, **options}
    last_error: Exception | None = None
    for kwargs in connection_attempts(model):
        try:
            chunks = (
                chunk.choices[0].delta.content or ""
                for chunk in litellm.completion(messages=messages, stream=True, **options, **kwargs)
            )
            first = next((text for text in chunks if text), "")
        except Exception as e:
            if not is_transient(e):
                raise LLMError(str(e), e) from e
            last_error = e
            continue
        return _prepend(first, chunks)
    raise LLMUnavailableError(str(last_error), last_error)


def _prepend(first: str, rest: Iterator[str]) -> Iterator[str]:
    if first:
        yield first
    for text in rest:
        if text:
            yield text
