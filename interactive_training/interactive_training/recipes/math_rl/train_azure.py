"""Math RL training using Azure AI Fine-Tuning Sessions SDK.

Near-identical to train.py (the direct base_url variant, no Azure identity required).
The differences are:
  1. ``base_url`` → ``project_endpoint`` (CLIConfig field rename)
  2. Auth + ``FineTuningSession`` creation inserted at the top of ``cli_main()``
  3. Calls ``rl.train_azure.main(config, session)`` instead of
     ``rl.train.main(config)``

Everything else — CLIConfig fields, ``get_dataset_builder()``, Config assembly,
logging, log-dir handling — is identical to train.py so diffs are minimal.

Usage::

    python -m interactive_training.recipes.math_rl.train_azure \\
        project_endpoint="https://..." \\
        model_name="Qwen/Qwen3.8-27B" env=gsm8k \\
        learning_rate=2e-5 temperature=1.0 max_tokens=1200 lora_rank=16 \\
        group_size=8 groups_per_batch=256 loss_fn=importance_sampling seed=42 \\
        eval_every=999999 save_every=999999 behavior_if_log_dir_exists=delete
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import chz
import torch
from azure.ai.finetuningsessions.models import Datum, LossFn as LossFnType, TensorData

from interactive_training import cli_utils, model_info
from interactive_training.dynamic_batching import build_strategy
from interactive_training.recipes.math_rl import arithmetic_env, math_env
from interactive_training.rl import train_azure as rl_azure
from interactive_training.rl.train import (
    AsyncConfig,
    Config,
    EvaluationStrategy,
    StreamMinibatchConfig,
)
from interactive_training.rl.train_azure import AzureSDKTrainingClient
from interactive_training.rl.types import RLDatasetBuilder
from interactive_training.tensor_utils import tensor_data_to_torch
from interactive_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from interactive_training.utils import file_utils

logger = logging.getLogger(__name__)

_DEFAULT_MATH_RL_LORA_RANK = 16
_DEFAULT_MATH_RL_LORA_ALPHA = 32


# =============================================================================
# Custom losses -- opt-in via --custom_loss=<name>
#
# When set, the recipe routes every ``forward_backward_async`` call through
# ``forward_backward_custom_async`` with a Python-side loss closure.  This is
# the user-facing extension point for shipping a research loss without waiting
# for a backend release.  See ``tests/test_surrogate_gradient_equivalence.py``
# for educational, side-by-side correctness proofs of the same algorithm.
#
# Supported selectors:
#   "is_py"             -- vanilla importance sampling, hand-rolled in Python.
#                          Should produce reward curves indistinguishable from
#                          the builtin ``importance_sampling`` loss.  Use this
#                          as the basis for your own RL loss variants.
#   "squared_logprob"   -- confidence-squared regulariser
#                          ``C = sum_t logprob_t**2``.  Reward-agnostic
#                          (no advantages term), so this reinforces every
#                          sampled token equally -- effectively SFT on the
#                          model's own rollouts with a confidence-emphasising
#                          weight, not RL.  Its primary use here is a fast
#                          end-to-end check that the surrogate machinery is
#                          healthy against your real session; whether it
#                          incidentally improves accuracy depends on whether
#                          the base model's rollouts are net-positive.
#   "dapo"              -- DAPO decoupled-clip surrogate
#                          (Yu et al., arXiv:2503.14476).  Token-level PPO
#                          objective with an asymmetric clip range
#                          ``[1 - eps_low, 1 + eps_high]`` -- the higher
#                          upper bound ("clip-higher") gives positive-
#                          advantage tokens more headroom to climb, which
#                          the paper reports reduces entropy collapse on
#                          reasoning tasks.  Hardcoded to the paper's
#                          defaults ``eps_low=0.2, eps_high=0.28``.  This
#                          is the canonical example of a loss that needs
#                          *two* per-token aux quantities (advantages and
#                          sampling logprobs) -- packed into separate
#                          ``loss_fn_inputs`` keys, never reaching the
#                          server (see docs/loss_functions.md).
# =============================================================================

_SUPPORTED_CUSTOM_LOSSES = ("is_py", "squared_logprob", "dapo")

# DAPO clip thresholds from the paper (arXiv:2503.14476).
_DAPO_EPS_LOW = 0.2
_DAPO_EPS_HIGH = 0.28


def _repack_for_is(d: Datum) -> Datum:
    """Pre-compute the IS multiplier on the client and pack into ``weights``.

    The standard IS loss is

        C = -sum_t mask_t * adv_t * exp(lp_t - lp_sampling_t)

    To carry both ``adv`` and ``lp_sampling`` (and ``mask``) through the
    single ``weights`` aux field, we precompute

        w_t = mask_t * adv_t * exp(-lp_sampling_t)

    and the Python loss becomes the clean one-liner

        C = -sum_t w_t * exp(lp_t).

    Mathematically equivalent to the original objective.
    """
    adv = torch.tensor(d.loss_fn_inputs["advantages"].data, dtype=torch.float64)
    samp_lp = torch.tensor(d.loss_fn_inputs["logprobs"].data, dtype=torch.float64)
    mask = torch.tensor(d.loss_fn_inputs["mask"].data, dtype=torch.float64)
    is_multiplier = mask * adv * torch.exp(-samp_lp)
    return Datum(
        model_input=d.model_input,
        loss_fn_inputs={
            "target_tokens": d.loss_fn_inputs["target_tokens"],
            "weights": TensorData(data=is_multiplier.to(torch.float32).tolist()),
        },
    )


def _is_py_loss(data, logprobs_list):
    """Vanilla importance-sampling loss: ``C = -sum_t w_t * exp(lp_t)``.

    ``w_t`` carries the precomputed ``mask * adv * exp(-lp_sampling)`` from
    :func:`_repack_for_is`; ``lp_t`` is the current model's per-token
    logprob (a leaf tensor with ``requires_grad=True``, provided by the
    adapter).
    """
    total = torch.zeros((), dtype=torch.float32)
    n_tokens = 0
    for d, lp in zip(data, logprobs_list, strict=True):
        w = tensor_data_to_torch(d.loss_fn_inputs["weights"])
        total = total + -(w * torch.exp(lp)).sum()
        n_tokens += int(lp.numel())
    metrics = {
        "custom_loss/is_py_total": float(total.detach().item()),
        "custom_loss/is_py_per_token": float(total.detach().item()) / max(n_tokens, 1),
    }
    return total, metrics


def _repack_for_squared(d: Datum) -> Datum:
    """Strip aux keys; squared-logprob doesn't use ``weights``."""
    return Datum(
        model_input=d.model_input,
        loss_fn_inputs={"target_tokens": d.loss_fn_inputs["target_tokens"]},
    )


def _squared_logprob_loss(data, logprobs_list):
    """Confidence-squared regulariser ``C = sum_t lp_t**2``.

    Reward-agnostic (no advantages); reinforces every sampled token equally
    -- effectively SFT on the model's own rollouts with a confidence-
    emphasising weight, not RL.  Primarily a wire-path smoke test.
    """
    total = torch.zeros((), dtype=torch.float32)
    for lp in logprobs_list:
        total = total + (lp ** 2).sum()
    return total, {"custom_loss/squared_logprob": float(total.detach().item())}


_CUSTOM_LOSS_REGISTRY: dict[str, tuple[Any, Any]] = {
    # name -> (repack_fn, loss_fn)
    "is_py": (_repack_for_is, _is_py_loss),
    "squared_logprob": (_repack_for_squared, _squared_logprob_loss),
    "dapo": (None, None),  # filled in below after the helpers are defined
}


def _repack_for_dapo(d: Datum) -> Datum:
    """Pack the two DAPO per-token aux quantities into named keys.

    Unlike :func:`_repack_for_is` we cannot fold advantages and sampling
    logprobs into the single ``weights`` field, because DAPO's clip uses
    the *raw* ratio ``r = exp(lp - lp_sampling)`` and the *raw* advantage
    sign.  Instead we carry them as separate aux fields:

        loss_fn_inputs = {
            "target_tokens":      <unchanged>,
            "dapo/mask_adv":      mask_t * adv_t,        # zeros out prompt
            "dapo/sampling_lp":   lp_sampling_t,         # raw, no masking
        }

    These extra keys are visible inside :func:`_dapo_loss` (which receives
    the *original* data list) but the cookbook adapter strips them from
    the server-bound forward / surrogate-backward payloads, so the wire
    contract stays ``{target_tokens, weights}``.
    """
    adv = torch.tensor(d.loss_fn_inputs["advantages"].data, dtype=torch.float64)
    samp_lp = torch.tensor(d.loss_fn_inputs["logprobs"].data, dtype=torch.float64)
    mask = torch.tensor(d.loss_fn_inputs["mask"].data, dtype=torch.float64)
    mask_adv = (mask * adv).to(torch.float32)
    return Datum(
        model_input=d.model_input,
        loss_fn_inputs={
            "target_tokens": d.loss_fn_inputs["target_tokens"],
            "dapo/mask_adv": TensorData(data=mask_adv.tolist()),
            "dapo/sampling_lp": TensorData(data=samp_lp.to(torch.float32).tolist()),
        },
    )


def _dapo_loss(data, logprobs_list):
    """DAPO decoupled-clip surrogate (arXiv:2503.14476).

    Per token::

        r_t      = exp(lp_t - lp_sampling_t)
        surr1_t  = r_t * (mask_t * adv_t)
        surr2_t  = clamp(r_t, 1 - eps_low, 1 + eps_high) * (mask_t * adv_t)
        C        = -sum_t min(surr1_t, surr2_t)

    Token-level aggregation (sum, not trajectory mean) matches the paper's
    formulation.  Tokens with ``mask_t == 0`` (prompt, padding) contribute
    nothing.  We log clip fractions on each side -- these are the standard
    DAPO debugging signal for tuning ``eps_high``.
    """
    eps_low = _DAPO_EPS_LOW
    eps_high = _DAPO_EPS_HIGH
    total = torch.zeros((), dtype=torch.float32)
    n_active = 0  # tokens with mask_adv != 0
    n_clip_low = 0  # tokens where surr2 < surr1 due to lower bound
    n_clip_high = 0  # tokens where surr2 < surr1 due to upper bound
    for d, lp in zip(data, logprobs_list, strict=True):
        mask_adv = tensor_data_to_torch(d.loss_fn_inputs["dapo/mask_adv"])
        samp_lp = tensor_data_to_torch(d.loss_fn_inputs["dapo/sampling_lp"])
        r = torch.exp(lp - samp_lp)
        r_clipped = torch.clamp(r, min=1.0 - eps_low, max=1.0 + eps_high)
        surr1 = r * mask_adv
        surr2 = r_clipped * mask_adv
        total = total + -torch.minimum(surr1, surr2).sum()
        with torch.no_grad():
            active = mask_adv != 0
            n_active += int(active.sum().item())
            # surr2 < surr1 iff clip was binding AND on the harmful side of A_t.
            clipped = (surr2 < surr1) & active
            n_clip_low += int((clipped & (r < 1.0 - eps_low)).sum().item())
            n_clip_high += int((clipped & (r > 1.0 + eps_high)).sum().item())
    metrics = {
        "custom_loss/dapo_total": float(total.detach().item()),
        "custom_loss/dapo_per_token": float(total.detach().item())
        / max(n_active, 1),
        "custom_loss/dapo_clip_low_frac": n_clip_low / max(n_active, 1),
        "custom_loss/dapo_clip_high_frac": n_clip_high / max(n_active, 1),
    }
    return total, metrics


_CUSTOM_LOSS_REGISTRY["dapo"] = (_repack_for_dapo, _dapo_loss)


class CustomLossTrainingClient(AzureSDKTrainingClient):
    """Drop-in :class:`AzureSDKTrainingClient` that routes every
    ``forward_backward_async`` through ``forward_backward_custom_async``.

    The shared ``train_step`` in :mod:`interactive_training.rl.train` calls
    ``training_client.forward_backward_async(batch, loss_fn=cfg.loss_fn, ...)``.
    By overriding that single method we keep the entire RL training loop
    untouched while swapping in a Python loss.
    """

    def __init__(self, client, session_id, tokenizer, *, repack_fn, custom_loss_fn):
        super().__init__(client, session_id, tokenizer)
        self._repack_fn = repack_fn
        self._custom_loss_fn = custom_loss_fn
        self._warned_ignored_loss_args = False

    async def forward_backward_async(
        self, batch, *, loss_fn=None, loss_fn_config=None
    ):  # type: ignore[override]
        if (loss_fn is not None or loss_fn_config is not None) and not self._warned_ignored_loss_args:
            logger.info(
                "CustomLossTrainingClient ignoring caller-provided loss_fn=%r "
                "loss_fn_config=%r; routing through forward_backward_custom_async "
                "with the registered Python closure instead.",
                loss_fn,
                loss_fn_config,
            )
            self._warned_ignored_loss_args = True
        repacked = [self._repack_fn(d) for d in batch]
        return await self.forward_backward_custom_async(
            repacked, loss_fn=self._custom_loss_fn
        )


# =============================================================================
# CLI configuration — identical to train.py except base_url → project_endpoint
# =============================================================================


@chz.chz
class CLIConfig:
    """Command-line config for Azure SDK-based math RL training."""

    # Model
    model_name: str = model_info.DEFAULT_MODEL_NAME
    tokenizer_name: str | None = None
    lora_rank: int = _DEFAULT_MATH_RL_LORA_RANK
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment
    env: str = "arithmetic"
    seed: int = 0

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 5
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 20
    eval_strategy: EvaluationStrategy = "steps"
    save_every: int = 20

    # Azure SDK endpoint (replaces base_url from train.py)
    project_endpoint: str

    # Auth / SDK options
    verbose_http: bool = False
    create_session_timeout_sec: float = 600.0

    training_type: TrainingType | None = training_type_field()

    user_metadata: dict[str, Any] | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Cap on total training iterations (None = use full dataset).
    # A safety net against runaway runs that burn GPU-minutes.
    max_steps: int | None = None
    # Wall-clock budget in seconds (None = no limit). The training loop
    # breaks at the next iteration boundary once elapsed time exceeds this.
    max_wall_clock_seconds: float | None = None

    # Dataset truncation — set to None for a full run (7500 train / 5000 test).
    # Change to 1000 / 200 for a ~1 hr run.
    max_train_examples: int | None = None
    max_test_examples: int | None = None

    # s1-style budget forcing (Muennighoff et al., 2025). When forced_commit
    # is enabled, rollouts that run out of tokens on turn 1 without producing
    # a boxed answer get a short second turn instructing them to commit. Only
    # supported on env in {math, deepmath}; ignored elsewhere. See MathEnv.
    forced_commit: bool = False
    # Reward penalty applied when an answer came from the forced-commit turn.
    # Positive values let the rollout-time recovery still help training while
    # encouraging the policy to find the answer on its own turn 1.
    forced_commit_penalty: float = 0.0
    # Max tokens for the forced-commit turn-2 generation. Small (the model
    # only needs to write a boxed answer); keeps worst-case per-rollout
    # token count bounded at turn1_max + this value.
    forced_commit_max_tokens: int = 1024
    # When true, prefill ``</think>\n\n\boxed{`` into turn-2 so the model
    # cannot deliberate about formatting and simply completes the box.
    # Drastically shortens the turn-2 budget required.
    forced_commit_prefill_boxed: bool = False

    # (experimental) Dynamic sampling (DAPO-style): keep sampling new prompt
    # groups until groups_per_batch non-constant-reward groups are collected.
    # This ensures a consistent effective batch size as the model improves.
    dynamic_sampling: bool = False
    max_oversample_rounds: int = 10

    # (experimental) Dynamic-batching strategy. None = vanilla GRPO (baseline).
    # The name is resolved to a strategy object by
    # interactive_training.dynamic_batching.build_strategy; the *_ params below tune
    # the individual methods. max_rolls_per_group caps A-axis re-rolls (MORE);
    # it must be set for oversampling methods (pilot_commit) to grow a prompt's
    # pool beyond the initial wave.
    strategy: str | None = None
    max_rolls_per_group: int | None = None
    # Refill the batch to groups_per_batch when the strategy DROPs a group
    # (instead of letting the batch shrink). Makes drop-based methods (dapo,
    # pilot_commit) faithful to their papers. Only meaningful with `strategy`.
    refill_on_drop: bool = False
    pods_keep: int | None = None
    pilot_p_lower: float = 0.125
    pilot_p_upper: float = 0.75
    pilot_rollouts: int | None = None
    commit_rollouts: int = 0
    # Minibatch streaming: when set, run the streaming RL loop
    # (do_sync_training_with_stream_minibatch) with this many minibatches per
    # substep instead of the plain sync loop. None = no streaming. This exposes
    # the third execution mode (sync / async / streaming) end-to-end.
    stream_num_minibatches: int | None = None

    # Max concurrent group rollouts per step (caps in-flight HTTP requests).
    # Each group spawns group_size HTTP requests; with 256 groups × 8 samples
    # that's 2048 concurrent requests.  None = unlimited; lower this if the
    # service or your local environment cannot sustain the request rate.
    max_concurrent_groups: int | None = None

    # Opt-in Python-side custom loss.  When set, every ``forward_backward``
    # call is routed through ``forward_backward_custom_async`` with a
    # hand-rolled loss closure (see ``_CUSTOM_LOSS_REGISTRY``).  Set to one of:
    #   - "is_py"           : vanilla IS, hand-rolled (drives learning;
    #                         compare reward curves against the builtin
    #                         ``importance_sampling`` for an E2E correctness
    #                         check).
    #   - "squared_logprob" : confidence-squared regulariser
    #                         ``C = sum_t logprob_t**2``; reward-agnostic
    #                         (no advantages term), so this is SFT-on-own-
    #                         rollouts with a confidence-emphasising weight
    #                         -- primarily a smoke test for the wire path.
    #   - "dapo"            : DAPO decoupled-clip surrogate
    #                         (arXiv:2503.14476).  Asymmetric PPO clip with
    #                         eps_low=0.20, eps_high=0.28.  Canonical example
    #                         of a custom loss that needs *two* per-token
    #                         aux quantities (advantages + sampling logprobs)
    #                         -- see docs/loss_functions.md "multi-aux-key
    #                         pattern".
    # ``loss_fn`` is ignored when ``custom_loss`` is set.
    custom_loss: str | None = None

    # --- Rollout resilience (advanced; opt-in) ---
    # Per-trajectory retry budget (0 disables retries).
    max_retries_per_trajectory: int = 0
    # Extra whole-group attempts allowed to backfill failed trajectory slots.
    max_extra_trajectory_attempts_per_group: int = 0


# =============================================================================
# Dataset builder — identical to train.py.
# =============================================================================


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    max_train_examples: int | None = None,
    max_test_examples: int | None = None,
    forced_commit: bool = False,
    forced_commit_penalty: float = 0.0,
    forced_commit_max_tokens: int = 1024,
    forced_commit_prefill_boxed: bool = False,
) -> RLDatasetBuilder:
    if env == "arithmetic":
        return arithmetic_env.ArithmeticDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            n_batches=100,
            include_fewshot=True,
            group_size=group_size,
        )
    elif env in ["math", "polaris", "deepmath", "gsm8k", "gsm1k", "gsm100"]:
        return math_env.get_math_dataset_builder(
            dataset_name=env,
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
            forced_commit=forced_commit,
            forced_commit_penalty=forced_commit_penalty,
            forced_commit_max_tokens=forced_commit_max_tokens,
            forced_commit_prefill_boxed=forced_commit_prefill_boxed,
        )
    else:
        raise ValueError(f"Unknown environment: {env}")


# =============================================================================
# cli_main — mirrors train.py cli_main() then creates session + calls rl_azure
# =============================================================================


def _parse_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
    """Parse a checkpoint path into ``(session_id, checkpoint_name)``.

    Accepts ``<session_id>/<name>`` URIs (as written to
    checkpoints.jsonl) or plain ``<session_id>/<checkpoint_name>`` paths.

    Raises ``ValueError`` on any other shape.
    """
    # Accept SDK-returned checkpoint URIs as well as session/checkpoint paths.
    if "://" in checkpoint_path:
        stripped = checkpoint_path.split("://", 1)[1]
        parts = stripped.split("/")
        if len(parts) >= 2:
            session = parts[0]
            name = parts[-1]
            if not session.startswith("session_"):
                session = f"session_{session.removeprefix('model_')}"
            return session, name

    parts = checkpoint_path.split("/", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        session = parts[0]
        if not session.startswith("session_"):
            session = f"session_{session.removeprefix('model_')}"
        return session, parts[1]

    raise ValueError(
        f"Invalid checkpoint_path={checkpoint_path!r}. "
        f"Expected format '<session_id>/<checkpoint_name>' (e.g. 'session_abc12345/final')."
    )


async def cli_main(cli_config: CLIConfig):
    """Build Config (same as train.py), create Azure session, run training."""

    cli_config_dump = {
        key: getattr(cli_config, key, None) for key in CLIConfig.__annotations__.keys()
    }
    print(
        f"[train_azure] cli_config:\n{json.dumps(cli_config_dump, indent=2, sort_keys=True, default=str)}"
    )

    # ── Same logic as train.py cli_main() ────────────────────────────────────
    tokenizer_name = cli_config.tokenizer_name or cli_config.model_name

    effective_lora_rank = cli_config.lora_rank
    effective_lora_alpha = _DEFAULT_MATH_RL_LORA_ALPHA
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name_safe = cli_config.model_name.replace("/", "-")
    loss_fn_str = getattr(cli_config.loss_fn, "value", cli_config.loss_fn)
    # Resolve the effective Config loss fields up front so Config and
    # run_meta share a single source of truth (Config).  When a Python-side
    # custom loss is selected, cli_config.loss_fn is ignored at runtime.
    if cli_config.custom_loss is not None:
        config_loss_fn: str = "custom"
        config_custom_loss_name: str | None = cli_config.custom_loss
        loss_label = cli_config.custom_loss
    else:
        config_loss_fn = cli_config.loss_fn
        config_custom_loss_name = None
        loss_label = loss_fn_str
    run_name = (
        f"{cli_config.env}-azure-sdk-{model_name_safe}"
        f"-{effective_lora_rank}rank"
        f"-{cli_config.learning_rate}lr"
        f"-{cli_config.group_size}group"
        f"-{cli_config.groups_per_batch}batch"
        f"-{loss_label}"
        f"-seed{cli_config.seed}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    log_path = (
        cli_config.log_path
        if cli_config.log_path is not None
        else str(file_utils.default_logs_root() / "math_rl" / run_name)
    )
    wandb_name = cli_config.wandb_name or run_name

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name=tokenizer_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
            max_train_examples=cli_config.max_train_examples,
            max_test_examples=cli_config.max_test_examples,
            forced_commit=cli_config.forced_commit,
            forced_commit_penalty=cli_config.forced_commit_penalty,
            forced_commit_max_tokens=cli_config.forced_commit_max_tokens,
            forced_commit_prefill_boxed=cli_config.forced_commit_prefill_boxed,
        ),
        model_name=tokenizer_name,
        tokenizer_name=tokenizer_name,
        lora_rank=effective_lora_rank,
        lora_alpha=effective_lora_alpha,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_strategy=cli_config.eval_strategy,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        compute_post_kl=cli_config.compute_post_kl,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=cli_config.groups_per_batch,
            num_minibatches=cli_config.stream_num_minibatches,
        )
        if cli_config.stream_num_minibatches is not None
        else None,
        loss_fn=config_loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        custom_loss_name=config_custom_loss_name,
        dynamic_sampling=cli_config.dynamic_sampling,
        max_oversample_rounds=cli_config.max_oversample_rounds,
        remove_constant_reward_groups=cli_config.dynamic_sampling,
        max_concurrent_groups=cli_config.max_concurrent_groups,
        strategy=build_strategy(
            cli_config.strategy,
            group_size=cli_config.group_size,
            pods_keep=cli_config.pods_keep,
            pilot_p_lower=cli_config.pilot_p_lower,
            pilot_p_upper=cli_config.pilot_p_upper,
            pilot_rollouts=cli_config.pilot_rollouts,
            commit_rollouts=cli_config.commit_rollouts,
        ),
        max_rolls_per_group=cli_config.max_rolls_per_group,
        refill_on_drop=cli_config.refill_on_drop,
        max_steps=cli_config.max_steps,
        max_wall_clock_seconds=cli_config.max_wall_clock_seconds,
        sampling_seed=cli_config.seed,
        max_retries_per_trajectory=cli_config.max_retries_per_trajectory,
        max_extra_trajectory_attempts_per_group=cli_config.max_extra_trajectory_attempts_per_group,
    )

    print(f"[train_azure] resolved training config: {config!r}")

    cli_utils.check_log_dir(
        log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    os.makedirs(log_path, exist_ok=True)

    # ── Write run_meta.json: surfaces context the dashboard can show ─────────
    parsed_endpoint = urlparse(cli_config.project_endpoint or "")
    endpoint_host = parsed_endpoint.netloc or cli_config.project_endpoint or ""
    host_parts = endpoint_host.split(".")

    # Heuristic project name from endpoint
    # e.g. "myproj.eastus2.api.azureml.ms" → "myproj"
    # or path ".../api/projects/<name>" → <name>
    endpoint_project: str | None = None
    import re as _re
    pm = _re.search(r"/projects/([^/?#]+)", parsed_endpoint.path or "")
    if pm:
        endpoint_project = pm.group(1)
    elif host_parts and host_parts[0] and host_parts[0].lower() != "www":
        endpoint_project = host_parts[0]

    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "log_path": log_path,
        "recipe": "math_rl_azure_sdk",
        "env": cli_config.env,
        "azure": {
            "project_endpoint": cli_config.project_endpoint,
            "project_name": endpoint_project,
            "endpoint_host": endpoint_host,
            "auth": "AzureKeyCredential" if os.environ.get("AZURE_AI_API_KEY") else "DefaultAzureCredential",
            "session_id": None,  # filled after session creation
            "reference_session_id": None,
            "from_checkpoint": None,  # filled below if continual fine-tuning
        },
        "model": {
            "model_name": cli_config.model_name,
            "tokenizer_name": tokenizer_name,
            "renderer_name": renderer_name,
            "lora_rank": effective_lora_rank,
            "lora_alpha": effective_lora_alpha,
        },
        "dataset": {
            "name": cli_config.env,
            "max_train_examples": cli_config.max_train_examples,
            "max_test_examples": cli_config.max_test_examples,
            "group_size": cli_config.group_size,
            "groups_per_batch": cli_config.groups_per_batch,
        },
        "training": {
            "learning_rate": cli_config.learning_rate,
            "temperature": cli_config.temperature,
            "max_tokens": cli_config.max_tokens,
            "loss_fn": loss_label,
            "seed": cli_config.seed,
            "eval_strategy": cli_config.eval_strategy,
            "eval_every": cli_config.eval_every,
            "save_every": cli_config.save_every,
            "strategy": cli_config.strategy or "baseline",
            "max_rolls_per_group": cli_config.max_rolls_per_group,
            "refill_on_drop": cli_config.refill_on_drop,
        },
        "command": " ".join(sys.argv),
    }

    def _write_meta() -> None:
        with open(os.path.join(log_path, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, default=str)

    _write_meta()

    def record_reference_session(session_id: str) -> None:
        run_meta["azure"]["reference_session_id"] = session_id
        _write_meta()

    # ── Azure SDK: auth + session creation ───────────────────────────────────
    from azure.ai.finetuningsessions.aio import FineTuningSessionClient
    from azure.ai.finetuningsessions.models import FromCheckpoint, LoRAConfig

    azure_api_key = os.environ.get("AZURE_AI_API_KEY")
    if azure_api_key:
        from azure.core.credentials import AzureKeyCredential

        credential = AzureKeyCredential(azure_api_key)
        logger.info("Using AzureKeyCredential (AZURE_AI_API_KEY).")
    else:
        from azure.identity.aio import DefaultAzureCredential

        # MI is included so AML compute jobs can auth via the attached UAMI;
        # locally DAC falls through MI → CLI in the normal order. Set
        # AZURE_CLIENT_ID on AML jobs to pin which UAMI is used.
        credential = DefaultAzureCredential()
        logger.info(
            "Using DefaultAzureCredential (MI → CLI fallback; pin AZURE_CLIENT_ID for a specific UAMI)."
        )

    if cli_config.verbose_http:
        import azure.ai.finetuningsessions._patch as _patch_module

        _patch_module.VERBOSE_HTTP = True
        logging.getLogger("azure.ai.finetuningsessions._patch").setLevel(logging.INFO)

    client_kwargs: dict[str, Any] = dict(
        endpoint=cli_config.project_endpoint,
        credential=credential,
    )
    if not azure_api_key:
        client_kwargs["credential_scopes"] = ["https://ai.azure.com/.default"]

    client = FineTuningSessionClient(**client_kwargs)

    # ── Determine if we're resuming from a checkpoint ────────────────────────
    from interactive_training import checkpoint_utils

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    selection = checkpoint_utils.select_resume_checkpoint(
        resume_info, cli_config.load_checkpoint_path
    )
    checkpoint_path = selection.checkpoint_path
    if selection.ignored_load_checkpoint_path:
        # Auto-resume preempts an explicit load_checkpoint_path so the restored
        # weights stay consistent with the dataset cursor read from this ledger.
        print(
            f"Ignoring load_checkpoint_path={selection.ignored_load_checkpoint_path}: "
            f"resuming from the existing ledger in {log_path} instead. Use a fresh "
            "log_path to continual-fine-tune from an explicit checkpoint."
        )
    if selection.is_resume:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    elif checkpoint_path is not None:
        print(f"Continual fine-tuning from checkpoint: {checkpoint_path}")

    from_checkpoint: FromCheckpoint | None = None

    if checkpoint_path is not None:
        source_session_id, checkpoint_id = _parse_checkpoint_path(checkpoint_path)
        from_checkpoint = FromCheckpoint(
            source_session_id=source_session_id,
            checkpoint_id=checkpoint_id,
        )
        run_meta["azure"]["from_checkpoint"] = {
            "source_session_id": source_session_id,
            "checkpoint_id": checkpoint_id,
        }
        _write_meta()

    print(
        f"Creating session: base_model={cli_config.model_name}  "
        f"lora_rank={effective_lora_rank}  lora_alpha={effective_lora_alpha}  "
        f"endpoint={cli_config.project_endpoint}"
        + (f"  from_checkpoint={from_checkpoint}" if from_checkpoint else "")
    )

    lora_config = LoRAConfig(rank=effective_lora_rank)
    try:
        session_id = await client.create_session(
            base_model=cli_config.model_name,
            lora_config=lora_config,
            type="training",
            from_checkpoint=from_checkpoint,
            timeout_sec=cli_config.create_session_timeout_sec,
            user_metadata=cli_config.user_metadata,
            training_type=normalize_training_type(cli_config.training_type),
        )
        print(f"Session ready: session_id={session_id}")

        run_meta["azure"]["session_id"] = session_id
        _write_meta()

        # ── Run training using the Azure SDK backend ──────────────────────────
        training_client_factory = None
        if cli_config.custom_loss is not None:
            if cli_config.custom_loss not in _CUSTOM_LOSS_REGISTRY:
                raise ValueError(
                    f"Unknown custom_loss={cli_config.custom_loss!r}.  "
                    f"Supported: {_SUPPORTED_CUSTOM_LOSSES}."
                )
            repack_fn, custom_loss_fn = _CUSTOM_LOSS_REGISTRY[cli_config.custom_loss]
            logger.info(
                "Custom loss enabled: %s -- forward_backward routed through "
                "forward_backward_custom_async (cli_config.loss_fn=%s is ignored).",
                cli_config.custom_loss,
                loss_fn_str,
            )

            def training_client_factory(client, session_id, tokenizer):
                return CustomLossTrainingClient(
                    client,
                    session_id,
                    tokenizer,
                    repack_fn=repack_fn,
                    custom_loss_fn=custom_loss_fn,
                )

        await rl_azure.main(
            config,
            client,
            session_id,
            training_type=cli_config.training_type,
            training_client_factory=training_client_factory,
            on_reference_session_created=record_reference_session,
        )
    finally:
        await client.close()
        if hasattr(credential, "close"):
            await credential.close()


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_utils.configure_logging(verbose_http=cli_config.verbose_http)
    asyncio.run(cli_utils.run_recipe(lambda: cli_main(cli_config)))
