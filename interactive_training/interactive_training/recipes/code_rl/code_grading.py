"""
Code grading utilities for RL training.

Supports two execution backends:
- sandboxfusion: Local Docker-based sandbox (default)
- modal: Cloud-based Modal sandbox
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from interactive_training.recipes.code_rl.lcb_utils import TEST_CODE, TEST_UTIL
from interactive_training.sandbox import SandboxBackend, SandboxFusionClient

logger = logging.getLogger(__name__)

# Substrings (lower-cased) in a sandbox error that mean the sandbox itself is
# unreachable or misconfigured, as opposed to the submitted code legitimately
# failing its tests. When the sandbox is down, every rollout silently grades as
# correct=0, which corrupts the RL reward signal, so we surface these loudly.
_SANDBOX_INFRA_ERROR_MARKERS = (
    "cannot connect to host",
    "connection refused",
    "connect call failed",
    "connectionreseterror",
    "connection reset",
    "max retries exceeded",
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
    "timeout",
)


class SandboxUnavailableError(RuntimeError):
    """Raised when the grading sandbox is persistently unreachable mid-run.

    code_rl grades rewards by executing code in a sandbox. If the sandbox dies
    after startup, every rollout grades as correct=0 and training produces no
    learning signal. This error is raised to abort the run instead of silently
    burning compute; it is re-raised (not swallowed) by the reward function and
    the check_solution tool so it propagates up and stops training.
    """


class _SandboxHealthTracker:
    """Tracks consecutive sandbox infrastructure failures across grading calls.

    A single transient blip should not kill a long run, but a sandbox that is
    truly down produces a continuous stream of infra failures. Once the number
    of *consecutive* infra failures crosses a threshold we treat the sandbox as
    down and abort. Any successful execution (even one whose tests fail) resets
    the counter. Rollouts run on a single asyncio event loop, so the plain
    counter needs no locking.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self.consecutive_failures = 0

    def record_infra_failure(self) -> int:
        self.consecutive_failures += 1
        return self.consecutive_failures

    def record_success(self) -> None:
        self.consecutive_failures = 0


# Abort the run after this many consecutive infra failures (env-overridable).
_SANDBOX_MAX_CONSECUTIVE_INFRA_FAILURES = int(
    os.getenv("SANDBOX_MAX_CONSECUTIVE_INFRA_FAILURES", "8")
)
_sandbox_health = _SandboxHealthTracker(_SANDBOX_MAX_CONSECUTIVE_INFRA_FAILURES)

# Global sandbox backend clients (lazily initialized)
_sandboxfusion_client: SandboxFusionClient | None = None
_modal_pool: Any = None  # ModalSandboxPool, but avoid import at module level


def _get_sandboxfusion_client() -> SandboxFusionClient:
    """Get or create the SandboxFusion client."""
    global _sandboxfusion_client
    if _sandboxfusion_client is None:
        _sandboxfusion_client = SandboxFusionClient()
    return _sandboxfusion_client


def _get_modal_pool():
    """Get or create the Modal sandbox pool."""
    global _modal_pool
    if _modal_pool is None:
        import modal

        from interactive_training.sandbox.modal_sandbox import ModalSandboxPool

        image = modal.Image.debian_slim().pip_install("numpy")
        _modal_pool = ModalSandboxPool(image=image)
    return _modal_pool


def extract_code_from_model(model_response: str) -> str | None:
    """Extract the last fenced code block from a model response."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def postprocess_lcb_sample(sample: list[dict[str, Any]]) -> dict[str, str]:
    """Convert test cases to LiveCodeBench format for the test runner."""
    sample_inputs = [item["input"] for item in sample]
    sample_outputs = [item["output"] for item in sample]

    sample_dict: dict[str, Any] = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name")
        if fn_name is None:
            raise AssertionError(f"Function name missing in metadata: {metadata}. Sample: {sample}")
        sample_dict["fn_name"] = fn_name

    return {
        "input_output": json.dumps(sample_dict),
    }


async def _check_with_sandboxfusion(
    test_cases: dict[str, str],
    generation: str,
    timeout: int,
    total_timeout: int,
) -> tuple[bool, dict[str, Any]]:
    """Execute tests using SandboxFusion backend."""
    client = _get_sandboxfusion_client()

    return await client.run(
        code=TEST_CODE % {"timeout": timeout},
        files={
            "test_cases.txt": json.dumps(test_cases),
            "code.py": generation,
            "testing_util.py": TEST_UTIL,
        },
        timeout=total_timeout,
    )


async def _check_with_modal(
    test_cases: dict[str, str],
    generation: str,
    timeout: int,
    total_timeout: int,
) -> tuple[bool, dict[str, Any]]:
    """Execute tests using Modal sandbox."""
    pool = _get_modal_pool()
    exit_code, stdout, stderr = await pool.run_in_workdir(
        files={
            "test_cases.txt": json.dumps(test_cases),
            "code.py": generation,
            "testing_util.py": TEST_UTIL,
            "run.py": TEST_CODE % {"timeout": timeout},
        },
        command=["python", "run.py"],
        timeout=total_timeout,
    )
    return exit_code == 0, {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}


def _is_sandbox_infra_error(details: dict[str, Any]) -> bool:
    """Heuristically detect whether a failure is the sandbox being unreachable.

    Distinguishes infrastructure failures (sandbox down / wrong URL / network)
    from a submitted solution legitimately failing its test cases.
    """
    error = str(details.get("error", "")).lower()
    if not error:
        return False
    return any(marker in error for marker in _SANDBOX_INFRA_ERROR_MARKERS)


def _resolve_sandbox_endpoint(use_backend: SandboxBackend) -> str:
    """Best-effort description of where grading requests are being sent."""
    if use_backend == SandboxBackend.SANDBOXFUSION:
        return _get_sandboxfusion_client().url
    return str(use_backend)


async def assert_sandbox_reachable(backend: SandboxBackend | None = None) -> None:
    """Fail fast if the grading sandbox is not reachable.

    code_rl grades rewards by executing generated code in a sandbox. If the
    sandbox is unreachable, *every* rollout silently grades as correct=0,
    producing a flat, invalid reward signal that wastes an entire training run.
    Raising here turns that silent corruption into an immediate, actionable
    failure at startup.
    """
    use_backend = backend or SandboxBackend.SANDBOXFUSION
    if use_backend != SandboxBackend.SANDBOXFUSION:
        # Modal reachability is validated lazily on first use.
        logger.info("Skipping sandbox preflight for backend=%s", use_backend)
        return

    client = _get_sandboxfusion_client()
    endpoint = client.url
    try:
        ok, detail = await client.health_check()
    except Exception as e:
        ok, detail = False, {"error": str(e)}

    if not ok:
        raise RuntimeError(
            f"SandboxFusion is not reachable at {endpoint} (detail={detail}). "
            "code_rl grades rewards by executing code in this sandbox; without "
            "it every rollout scores correct=0 and the run produces no learning "
            "signal. Start SandboxFusion (e.g. `docker run -d -p 127.0.0.1:8080:8080 "
            "volcengine/sandbox-fusion:server-20250609`) or set SANDBOX_URL to a "
            "reachable endpoint before launching."
        )
    logger.info("SandboxFusion preflight OK at %s", endpoint)


async def sandbox_check_correctness(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int = 6,
    backend: SandboxBackend | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Check correctness of generated code using sandbox execution.

    Args:
        sample: List of test cases in LiveCodeBench format
        generation: Generated code to test
        timeout: Per-test timeout in seconds
        backend: Sandbox backend to use (defaults to "sandboxfusion")

    Returns:
        Tuple of (all_passed: bool, details: dict)
    """
    assert len(sample) >= 1, "Sample must contain at least one test case"

    # Process test cases
    test_cases = postprocess_lcb_sample(sample)
    use_backend = backend or SandboxBackend.SANDBOXFUSION

    try:
        test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
        total_timeout = (timeout + 1) * test_cnt + 5

        if use_backend == SandboxBackend.MODAL:
            passed, details = await _check_with_modal(
                test_cases, generation, timeout, total_timeout
            )
        elif use_backend == SandboxBackend.SANDBOXFUSION:
            passed, details = await _check_with_sandboxfusion(
                test_cases, generation, timeout, total_timeout
            )
        else:
            raise ValueError(f"Invalid sandbox backend: {use_backend}")
    except Exception as e:
        passed, details = False, {"error": str(e)}

    # Consumer-side logging of the EXACT output the sandbox returned for this
    # grade. Infra failures (sandbox unreachable) are logged loudly because they
    # silently zero the reward signal; normal results are logged at INFO.
    endpoint = _resolve_sandbox_endpoint(use_backend)
    if _is_sandbox_infra_error(details):
        consecutive = _sandbox_health.record_infra_failure()
        logger.error(
            "Sandbox UNREACHABLE/misconfigured (backend=%s endpoint=%s "
            "consecutive_infra_failures=%d/%d): %r. Code could not be executed, "
            "so this rollout is graded correct=0 and the reward signal is invalid "
            "until the sandbox is reachable.",
            use_backend,
            endpoint,
            consecutive,
            _sandbox_health.threshold,
            details,
        )
        if consecutive >= _sandbox_health.threshold:
            raise SandboxUnavailableError(
                f"SandboxFusion at {endpoint} failed {consecutive} consecutive "
                f"grading calls with infrastructure errors (last={details!r}). "
                "The sandbox appears to be down; aborting the run because every "
                "rollout now grades correct=0 and training produces no learning "
                "signal. Restore the sandbox (or set SANDBOX_URL to a reachable "
                "endpoint) and resume."
            )
    else:
        _sandbox_health.record_success()
        logger.info(
            "Sandbox result (backend=%s endpoint=%s): passed=%s details=%r",
            use_backend,
            endpoint,
            passed,
            details,
        )
    return passed, details


def taco_to_lcb_format(tests: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert TACO-style tests to LiveCodeBench format."""
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    n = max(len(inputs), len(outputs))

    test_cases: list[dict[str, Any]] = []
    for i in range(n):
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        if isinstance(out, list):
            out = out[0] if out else ""
        case: dict[str, Any] = {
            "input": inp,
            "output": out,
            "metadata": {},
        }
        if "fn_name" in tests:
            case["testtype"] = "functional"
            case["metadata"]["func_name"] = tests["fn_name"]
        else:
            case["testtype"] = "stdin_stdout"
        test_cases.append(case)

    return test_cases
