from types import SimpleNamespace
from unittest.mock import patch

import pytest
from litellm.types.llms.openai import ResponsesAPIResponse

from minisweagent.models.litellm_response_model import LitellmResponseModel


def _completed_tool_call_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-6-luna",
            "status": "completed",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": '{"command": "ls"}',
                    "status": "completed",
                }
            ],
        }
    )


def test_replayed_tool_call_uses_api_field_names():
    """A stored response must be replayed with the field names the Responses API accepts.

    Pydantic's default dump writes ``async_`` (the Python name for ``async``); the
    provider rejects that with HTTP 400 ``Unknown parameter: 'input[N].async_'``.
    """
    model = LitellmResponseModel(model_name="openai/test", cost_tracking="ignore_errors")
    events = iter([SimpleNamespace(type="response.completed", response=_completed_tool_call_response())])
    with patch("minisweagent.models.litellm_response_model.litellm.responses", return_value=events):
        message = model.query([{"role": "user", "content": "test"}])

    replayed = model._prepare_messages_for_api([{"role": "user", "content": "test"}, message])

    tool_call = replayed[1]
    assert tool_call["type"] == "function_call"
    assert "async_" not in tool_call
    assert tool_call["call_id"] == "call_1"
    assert tool_call["arguments"] == '{"command": "ls"}'
    assert message["extra"]["actions"][0]["command"] == "ls"


def test_replay_renames_python_field_names_in_previously_stored_messages():
    """Messages saved before the fix (e.g. a resumed trajectory) still contain ``async_``."""
    model = LitellmResponseModel(model_name="openai/test")
    stored = {
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "bash",
                "arguments": "{}",
                "async_": None,
                "extra": {"note": "local only"},
            }
        ],
    }

    (tool_call,) = model._prepare_messages_for_api([stored])

    assert tool_call == {"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": "{}", "async": None}


def test_query_streams_and_returns_completed_response():
    completed_response = SimpleNamespace(id="resp_123")
    events = iter(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta"),
            SimpleNamespace(type="response.completed", response=completed_response),
        ]
    )
    model = LitellmResponseModel(model_name="openai/test")

    with patch(
        "minisweagent.models.litellm_response_model.litellm.responses",
        return_value=events,
    ) as responses:
        result = model._query([{"role": "user", "content": "test"}])

    assert result is completed_response
    assert responses.call_args.kwargs["stream"] is True


def test_query_rejects_stream_without_completed_response():
    model = LitellmResponseModel(model_name="openai/test")

    with (
        patch(
            "minisweagent.models.litellm_response_model.litellm.responses",
            return_value=iter([SimpleNamespace(type="response.created")]),
        ),
        pytest.raises(RuntimeError, match="without a completed response"),
    ):
        model._query([{"role": "user", "content": "test"}])
