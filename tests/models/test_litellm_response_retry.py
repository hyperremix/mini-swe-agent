import httpx
import litellm
import pytest

from minisweagent.models.litellm_response_model import LitellmResponseModel


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr("tenacity.nap.time.sleep", sleeps.append)
    return sleeps


def _status_error(cls, status: int, headers: dict | None = None):
    request = httpx.Request("POST", "https://litellm.example/v1/responses")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls(message=f"HTTP {status}", llm_provider="openai", model="gpt-6-luna", response=response)


def test_rejected_request_is_not_retried(monkeypatch):
    """An HTTP 400 is caused by the request itself, so sending it again cannot succeed."""
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "8")
    model = LitellmResponseModel(model_name="openai/test", cost_tracking="ignore_errors")
    calls = []

    def reject(*args, **kwargs):
        calls.append(1)
        raise litellm.exceptions.BadRequestError(
            message="Unknown parameter: 'input[2].async_'", llm_provider="openai", model="gpt-6-luna"
        )

    monkeypatch.setattr(model, "_query", reject)

    with pytest.raises(litellm.exceptions.BadRequestError):
        model.query([{"role": "user", "content": "test"}])

    assert len(calls) == 1


def _fail_then_succeed(model, monkeypatch, error):
    calls = []
    completed = object()

    def query(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise error
        return completed

    monkeypatch.setattr(model, "_query", query)
    monkeypatch.setattr(model, "_calculate_cost", lambda response: {"cost": 0.0})
    monkeypatch.setattr(model, "_parse_actions", lambda response: [])
    monkeypatch.setattr("minisweagent.models.litellm_response_model._dump_response", lambda response: {})
    return calls


def test_rate_limit_waits_for_retry_after(monkeypatch, no_retry_sleep):
    model = LitellmResponseModel(model_name="openai/test", cost_tracking="ignore_errors")
    error = _status_error(litellm.exceptions.RateLimitError, 429, {"retry-after": "90"})
    calls = _fail_then_succeed(model, monkeypatch, error)

    model.query([{"role": "user", "content": "test"}])

    assert len(calls) == 2
    assert no_retry_sleep == [90]


def test_rate_limit_wrapped_by_litellm_stream_fallback_waits_longer(monkeypatch, no_retry_sleep):
    """LiteLLM reports a mid-stream Azure 429 as MidStreamFallbackError (a 503 subclass)."""
    model = LitellmResponseModel(model_name="openai/test", cost_tracking="ignore_errors")
    error = litellm.exceptions.MidStreamFallbackError(
        message="litellm.RateLimitError: AzureException RateLimitError - exceeded token rate limit",
        model="gpt-6-luna",
        llm_provider="openai",
        original_exception=Exception("rate limit"),
    )
    calls = _fail_then_succeed(model, monkeypatch, error)

    model.query([{"role": "user", "content": "test"}])

    assert len(calls) == 2
    assert no_retry_sleep[0] >= 30


def test_rate_limit_wait_is_capped(monkeypatch, no_retry_sleep):
    model = LitellmResponseModel(model_name="openai/test", cost_tracking="ignore_errors")
    error = _status_error(litellm.exceptions.RateLimitError, 429, {"retry-after": "86400"})
    _fail_then_succeed(model, monkeypatch, error)

    model.query([{"role": "user", "content": "test"}])

    assert no_retry_sleep == [300]


def test_server_error_keeps_short_exponential_backoff(monkeypatch, no_retry_sleep):
    model = LitellmResponseModel(model_name="openai/test", cost_tracking="ignore_errors")
    error = _status_error(litellm.exceptions.InternalServerError, 500)
    _fail_then_succeed(model, monkeypatch, error)

    model.query([{"role": "user", "content": "test"}])

    assert no_retry_sleep == [4]
