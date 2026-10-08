import asyncio
import http.client
import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar, overload

from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError
from azure.ai.finetuningsessions.models import ModelInput, ModelInputChunk

from interactive_training.completers import ContextLengthExceededError, TokenCompleter
from interactive_training.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from interactive_training.utils import logtree

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Max characters for log values in table cells before truncation
LOG_VALUE_MAX_LEN = 100


def _model_input_length(model_input: ModelInput) -> int:
    return sum(
        len(chunk.tokens) if isinstance(chunk, ModelInputChunk) else chunk.length
        for chunk in model_input.chunks
    )


class RolloutGroupExhausted(RuntimeError):
    """A rollout group could not fill every trajectory slot within its retry budget."""


class AuthoritativeGroupGradingError(RuntimeError):
    """An authoritative external grader could not score a completed rollout group."""


class IncompleteGroupGradingError(AuthoritativeGroupGradingError):
    """An exhausted, incomplete training reward vector permits a whole-group drop."""

    def __init__(self, message: str, *, trajectory_count: int) -> None:
        super().__init__(message)
        self.trajectory_count = trajectory_count


@dataclass(frozen=True)
class GroupDropEvent:
    """Training context supplied to an application's group-drop callback."""

    error: IncompleteGroupGradingError
    training_step: int | None
    additional_trajectories: int = 0

    @property
    def trajectory_count(self) -> int:
        return self.error.trajectory_count + self.additional_trajectories


GroupDropCallback = Callable[[GroupDropEvent], Awaitable[None] | None]


class RolloutFailureRateExceeded(RuntimeError):
    """Recent retryable rollout failures exceeded the configured circuit breaker."""


@dataclass(frozen=True)
class RolloutRetryPolicy:
    """Bounds how aggressively a rollout group retries failed trajectory slots."""

    max_retries_per_trajectory: int = 0
    max_extra_attempts_per_group: int = 0

    def __post_init__(self) -> None:
        if self.max_retries_per_trajectory < 0:
            raise ValueError("max_retries_per_trajectory must be non-negative")
        if self.max_extra_attempts_per_group < 0:
            raise ValueError("max_extra_attempts_per_group must be non-negative")


@dataclass
class RolloutFailureTracker:
    """Rolling-window circuit breaker over recent rollout attempt outcomes."""

    window_size: int = 100
    warn_rate: float = 0.10
    abort_rate: float = 0.20
    min_attempts: int = 25
    # When the breaker trips, allow this many recoveries (reset window + back off
    # + retry) before giving up. 0 = trip is fatal (original behavior).
    max_resets: int = 0
    # Pause this long after a trip to let the sampling backend recover before
    # retrying. Also used to fold near-simultaneous trips (e.g. from multiple
    # rollout workers) into a single logical recovery so they share one budget.
    reset_backoff_sec: float = 60.0
    _outcomes: deque[bool] = field(init=False)
    _warning_active: bool = field(default=False, init=False)
    _resets_used: int = field(default=0, init=False)
    _last_reset_monotonic: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0 <= self.warn_rate < self.abort_rate <= 1:
            raise ValueError("failure rates must satisfy 0 <= warn_rate < abort_rate <= 1")
        if self.min_attempts <= 0:
            raise ValueError("min_attempts must be positive")
        if self.max_resets < 0:
            raise ValueError("max_resets must be non-negative")
        if self.reset_backoff_sec < 0:
            raise ValueError("reset_backoff_sec must be non-negative")
        self._outcomes = deque(maxlen=self.window_size)

    @property
    def resets_used(self) -> int:
        return self._resets_used

    def reset(self) -> None:
        """Clear the rolling window so the breaker starts fresh."""
        self._outcomes.clear()
        self._warning_active = False

    def note_breaker_tripped(self) -> float | None:
        """Record that the breaker fired and decide whether to recover.

        Returns the number of seconds to back off before retrying, or ``None``
        when the reset budget is exhausted (the caller should abort).

        Trips that arrive within ``reset_backoff_sec`` of the last budget-
        consuming reset are treated as the same outage (e.g. a second rollout
        worker hitting the same degraded backend) and do not consume additional
        budget; they still reset the window and back off.
        """
        now = time.monotonic()
        same_outage = (
            self._last_reset_monotonic is not None
            and (now - self._last_reset_monotonic) < self.reset_backoff_sec
        )
        if not same_outage:
            if self._resets_used >= self.max_resets:
                return None
            self._resets_used += 1
            self._last_reset_monotonic = now
        self.reset()
        return self.reset_backoff_sec

    def record(self, *, successes: int, failures: int) -> None:
        self._outcomes.extend([False] * successes)
        self._outcomes.extend([True] * failures)
        if len(self._outcomes) < min(self.min_attempts, self.window_size):
            return

        failure_rate = sum(self._outcomes) / len(self._outcomes)
        if failure_rate >= self.abort_rate:
            raise RolloutFailureRateExceeded(
                f"rollout failure rate {failure_rate:.1%} over the last "
                f"{len(self._outcomes)} attempts exceeds {self.abort_rate:.1%}"
            )
        if failure_rate >= self.warn_rate and not self._warning_active:
            logger.warning(
                "Rollout failure rate is %.1f%% over the last %d attempts (warning threshold %.1f%%)",
                failure_rate * 100,
                len(self._outcomes),
                self.warn_rate * 100,
            )
            self._warning_active = True
        elif failure_rate < self.warn_rate:
            self._warning_active = False


def _is_retryable_rollout_error(error: BaseException) -> bool:
    """Only transient sampling/transport failures are safe to retry with a fresh slot."""
    if isinstance(
        error,
        (
            TimeoutError,
            ConnectionError,
            OSError,
            http.client.HTTPException,
            ServiceRequestError,
            ServiceResponseError,
        ),
    ):
        return True
    if isinstance(error, HttpResponseError):
        status_code = getattr(error, "status_code", None)
        if status_code is None and error.response is not None:
            status_code = error.response.status_code
        return status_code in (408, 429) or (status_code is not None and status_code >= 500)
    # The SDK surfaces some backend/transport sample failures as a plain
    # RuntimeError with the HTTP status embedded in the message (e.g. a
    # transient '403 Forbidden' or 5xx on a single replica), rather than an
    # HttpResponseError. Treat those sample-request failures as retryable so
    # they flow through the retry -> circuit-breaker path: a transient blip
    # self-heals into a fresh request/replica, and a genuinely persistent
    # failure trips the breaker (bounded) instead of crashing the run on the
    # first occurrence. ContextLengthExceededError is a RuntimeError subclass
    # with its own graceful handling, so it is explicitly excluded.
    if isinstance(error, RuntimeError) and not isinstance(error, ContextLengthExceededError):
        text = str(error)
        if "sample request" in text and "failed" in text:
            return True
    return False


async def _close_env(env: Env) -> None:
    try:
        await env.close()
    except Exception:
        logger.exception("Failed to close rollout environment")


def _truncate_log_value(value: Any, max_len: int = LOG_VALUE_MAX_LEN) -> tuple[str, bool]:
    """Truncate a log value if it's too long. Returns (display_value, was_truncated)."""
    str_value = str(value)
    if len(str_value) > max_len:
        return str_value[:max_len] + "...", True
    return str_value, False


@logtree.scope_header_decorator
async def do_single_rollout(
    policy: TokenCompleter,
    env: Env,
    selection_seed: int | None = None,
    *,
    close_on_success: bool = False,
) -> Trajectory:
    try:
        transitions = []
        ob, stop_condition = await env.initial_observation()
        max_tokens_override: int | None = None
        while True:
            try:
                ac_with_logprobs = await policy(ob, stop_condition, max_tokens_override)
            except ContextLengthExceededError:
                # The prompt alone has filled the model context window, so no
                # further action can be sampled. End the trajectory gracefully
                # with the transitions collected so far rather than crashing.
                # If we could not even take the first step there is nothing to
                # keep, so let it propagate to the group-level failure handling.
                if not transitions:
                    raise
                # By default, return the partial trajectory WITHOUT closing the
                # env: like the normal success path, the env must stay open so
                # do_group_rollout can read final state in compute_group_rewards
                # -- the group-level finally closes every env after rewards are
                # computed. close_on_success=True (see below) means the caller's
                # builder has confirmed it needs nothing further from this env.
                if close_on_success:
                    await _close_env(env)
                return Trajectory(
                    transitions=transitions,
                    final_ob=ob,
                    selection_seed=selection_seed,
                )
            step_result = await env.step(ac_with_logprobs.tokens)
            transition = Transition(
                ob=ob,
                ac=ac_with_logprobs,
                reward=step_result.reward,
                episode_done=step_result.episode_done,
                metrics=step_result.metrics,
                logs=step_result.logs,
            )
            transitions.append(transition)
            ob = step_result.next_observation
            stop_condition = step_result.next_stop_condition
            max_tokens_override = step_result.next_max_tokens
            if step_result.episode_done:
                break
        if close_on_success:
            # This rollout's builder (env_group_builder.env_needed_after_own_
            # trajectory() == False) has confirmed compute_group_rewards never
            # reads this env, so release it now instead of holding it for
            # the rest of the group. This lets builders release resources as
            # soon as their own trajectory finishes.
            await _close_env(env)
        return Trajectory(
            transitions=transitions,
            final_ob=ob,
            selection_seed=selection_seed,
        )
    except BaseException:
        await _close_env(env)
        raise


@overload
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
    retry_policy: RolloutRetryPolicy | None = None,
    failure_tracker: RolloutFailureTracker | None = None,
    result_mapper: None = None,
) -> TrajectoryGroup: ...


@overload
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
    retry_policy: RolloutRetryPolicy | None,
    failure_tracker: RolloutFailureTracker | None,
    result_mapper: Callable[[TrajectoryGroup, Sequence[Env]], T],
) -> tuple[TrajectoryGroup, T]: ...


@logtree.scope_header_decorator
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
    retry_policy: RolloutRetryPolicy | None = None,
    failure_tracker: RolloutFailureTracker | None = None,
    result_mapper: Callable[[TrajectoryGroup, Sequence[Env]], T] | None = None,
) -> TrajectoryGroup | tuple[TrajectoryGroup, T]:
    retries_enabled = retry_policy is not None
    retry_policy = retry_policy or RolloutRetryPolicy()
    envs_G = list(await env_group_builder.make_envs())
    selection_seeds = [secrets.randbits(32) for _ in envs_G]
    all_envs = list(envs_G)
    trajectories: list[Trajectory | None] = [None] * len(envs_G)
    retry_counts = [0] * len(envs_G)
    extra_attempts = 0
    pending_indices = list(range(len(envs_G)))
    close_on_success = not env_group_builder.env_needed_after_own_trajectory()

    try:
        while pending_indices:
            results = await asyncio.gather(
                *[
                    do_single_rollout(
                        policy,
                        envs_G[i],
                        selection_seeds[i],
                        close_on_success=close_on_success,
                    )
                    for i in pending_indices
                ],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException) and not isinstance(result, Exception):
                    raise result
            failures = [result for result in results if isinstance(result, BaseException)]
            if failure_tracker is not None:
                failure_tracker.record(
                    successes=len(results) - len(failures), failures=len(failures)
                )

            failed_indices: list[int] = []
            failed_errors: list[BaseException] = []
            for index, result in zip(pending_indices, results, strict=True):
                if isinstance(result, BaseException):
                    failed_indices.append(index)
                    failed_errors.append(result)
                else:
                    trajectories[index] = result

            if not failed_indices:
                break

            retryable = all(_is_retryable_rollout_error(error) for error in failed_errors)
            if not retryable:
                raise failed_errors[0]
            within_slot_budgets = all(
                retry_counts[index] < retry_policy.max_retries_per_trajectory
                for index in failed_indices
            )
            within_group_budget = (
                extra_attempts + len(failed_indices)
                <= retry_policy.max_extra_attempts_per_group
            )
            if not (within_slot_budgets and within_group_budget):
                if not retries_enabled:
                    raise failed_errors[0]
                raise RolloutGroupExhausted(
                    f"failed to fill {len(failed_indices)} of {len(envs_G)} trajectory slots "
                    f"after {extra_attempts} extra attempts"
                ) from failed_errors[0]

            replacement_envs = list(await env_group_builder.make_envs())
            if len(replacement_envs) != len(envs_G):
                raise RuntimeError(
                    "EnvGroupBuilder returned a different group size while retrying"
                )
            all_envs.extend(replacement_envs)
            for index in failed_indices:
                envs_G[index] = replacement_envs[index]
                retry_counts[index] += 1
            extra_attempts += len(failed_indices)
            pending_indices = failed_indices
            logger.warning(
                "Retrying %d/%d rollout trajectories (%d/%d extra attempts used)",
                len(failed_indices),
                len(envs_G),
                extra_attempts,
                retry_policy.max_extra_attempts_per_group,
            )

        trajectories_G = [trajectory for trajectory in trajectories if trajectory is not None]
        if len(trajectories_G) != len(envs_G):
            raise RuntimeError("rollout group completed with missing trajectories")
        rewards_and_metrics_G = await env_group_builder.compute_group_rewards(
            trajectories_G, envs_G
        )
        rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

        if extra_attempts and metrics_G:
            metrics_G[0]["rollout/retry_attempts"] = extra_attempts

        # Log trajectory tables with final rewards
        with logtree.scope_header("Trajectory Summary"):
            for i, (traj, final_reward) in enumerate(zip(trajectories_G, rewards_G, strict=True)):
                # Pre-scan to collect all log keys across all transitions (preserving order, deduped)
                all_log_keys = list(dict.fromkeys(key for t in traj.transitions for key in t.logs))

                rows = []
                truncated_values: list[tuple[int, str, str]] = []  # (step, key, full_value)
                step_reward_sum = 0.0
                for t_idx, t in enumerate(traj.transitions):
                    step_reward_sum += t.reward
                    row: dict[str, Any] = {
                        "step": t_idx,
                        "ob_len": _model_input_length(t.ob),
                        "ac_len": len(t.ac.tokens),
                        "reward": f"{t.reward:.3f}",
                    }
                    # Add log fields (user is responsible for avoiding collision with core columns)
                    for key in all_log_keys:
                        if key in t.logs:
                            display_val, was_truncated = _truncate_log_value(t.logs[key])
                            row[key] = display_val
                            if was_truncated:
                                truncated_values.append((t_idx, key, str(t.logs[key])))
                        else:
                            row[key] = "-"
                    rows.append(row)
                # Add final row with final observation and computed reward
                rows.append(
                    {
                        "step": "final",
                        "ob_len": _model_input_length(traj.final_ob),
                        "ac_len": "-",
                        "reward": f"{final_reward:.3f}",
                        **{key: "-" for key in all_log_keys},
                    }
                )
                # Add total reward row
                rows.append(
                    {
                        "step": "total",
                        "ob_len": "-",
                        "ac_len": "-",
                        "reward": f"{step_reward_sum + final_reward:.3f}",
                        **{key: "-" for key in all_log_keys},
                    }
                )
                logtree.table(rows, caption=f"Trajectory {i}")

                # Show full content for any truncated values in collapsible blocks
                for step_idx, key, full_value in truncated_values:
                    logtree.details(
                        full_value,
                        summary=f"Step {step_idx} - {key} (full, {len(full_value)} chars)",
                        pre=True,
                    )

        trajectory_group = TrajectoryGroup(
            trajectories_G,
            list(rewards_G),
            list(metrics_G),
            dynamic_batching_metadata=dict(
                env_group_builder.dynamic_batching_metadata()
            ),
        )
        if result_mapper is None:
            return trajectory_group
        return trajectory_group, result_mapper(trajectory_group, envs_G)
    finally:
        await asyncio.gather(*[_close_env(env) for env in all_envs])
