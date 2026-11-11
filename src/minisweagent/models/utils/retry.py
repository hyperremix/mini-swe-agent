"""Retry utility for model queries."""

import logging
import os

from tenacity import (
    RetryCallState,
    Retrying,
    before_sleep_log,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

RATE_LIMIT_MIN_WAIT_SEC = 30
RATE_LIMIT_MAX_WAIT_SEC = 300


def _is_rate_limit(exception: BaseException) -> bool:
    # Providers behind LiteLLM sometimes surface a 429 as another error type
    # (e.g. MidStreamFallbackError) whose message still names the rate limit.
    if getattr(exception, "status_code", None) == 429:
        return True
    return "ratelimit" in f"{type(exception).__name__} {exception}".lower().replace(" ", "").replace("_", "")


def _retry_after_sec(exception: BaseException) -> float | None:
    response = getattr(exception, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


class _wait_for_provider:
    """Exponential backoff, with longer waits for rate limits that honor Retry-After."""

    def __init__(self) -> None:
        self._default = wait_exponential(multiplier=1, min=4, max=60)

    def __call__(self, retry_state: RetryCallState) -> float:
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if exception is None or not _is_rate_limit(exception):
            return self._default(retry_state)
        retry_after = _retry_after_sec(exception)
        if retry_after is None:
            retry_after = RATE_LIMIT_MIN_WAIT_SEC * 2 ** (retry_state.attempt_number - 1)
        return min(RATE_LIMIT_MAX_WAIT_SEC, max(retry_after, self._default(retry_state)))


def retry(*, logger: logging.Logger, abort_exceptions: list[type[Exception]]) -> Retrying:
    """Thin wrapper around tenacity.Retrying to make use of global config etc.

    Args:
        logger: Logger to use for reporting retries
        abort_exceptions: Exceptions to abort on.

    Returns:
        A tenacity.Retrying object.
    """
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=_wait_for_provider(),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )
