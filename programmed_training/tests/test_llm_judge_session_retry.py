"""Session-allocation retry coverage for the BillSum judge recipe."""

from azure.ai.finetuningsessions import RequestRetryableError
from azure.core.exceptions import HttpResponseError, ServiceRequestError

from programmed_training.recipes.llm_judge_rl.train_azure import (
    _session_creation_retry_delay,
)


def test_session_creation_retry_classification_honors_sdk_hint():
    retryable = RequestRetryableError(
        "safe to retry",
        error_code="request_orphaned",
        retry_after_sec=7.5,
    )
    assert _session_creation_retry_delay(retryable, 0) == 7.5
    assert _session_creation_retry_delay(ServiceRequestError("disconnected"), 1) == 2

    timeout = HttpResponseError(message="timeout")
    timeout.status_code = 408
    assert _session_creation_retry_delay(timeout, 2) == 4

    fatal = HttpResponseError(message="bad request")
    fatal.status_code = 400
    assert _session_creation_retry_delay(fatal, 0) is None
