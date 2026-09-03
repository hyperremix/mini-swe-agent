from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisweagent.models.litellm_response_model import LitellmResponseModel


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
