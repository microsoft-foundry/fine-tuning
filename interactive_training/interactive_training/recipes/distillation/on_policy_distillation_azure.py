"""On-policy distillation using the Azure AI Fine-Tuning Sessions SDK.

Single-turn on-policy distillation: the student samples completions for Tulu3
prompts, the teacher (reference) model scores them, and the *only* training
signal is the KL penalty against the teacher (no correctness/format rewards).

This reuses the shared RL training loop (``interactive_training.rl.train``) via the
Azure backend (``interactive_training.rl.train_azure``); the teacher is wired in as the
KL reference session.

Smoke test (student == teacher == Qwen3.8-27B; the teacher does not need to be
good at any task — this just proves the code path):

    AZURE_AI_API_KEY=... uv run python -m \\
        interactive_training.recipes.distillation.on_policy_distillation_azure \\
        project_endpoint="https://..." \\
        model_name="Qwen/Qwen3.8-27B" \\
        kl_penalty_coef=1.0 group_size=2 groups_per_batch=2 \\
        max_train_examples=8 max_tokens=64 max_steps=2 \\
        eval_every=999999 save_every=999999 behavior_if_log_dir_exists=delete

Success = the run completes and logs a ``kl_policy_base`` metric.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any

import chz
from azure.ai.finetuningsessions.models import LossFn as LossFnType

from interactive_training import cli_utils, model_info
from interactive_training.distillation.datasets import TuluPromptDatasetBuilder
from interactive_training.rl import train_azure as rl_azure
from interactive_training.rl.train import Config, KLReferenceConfig
from interactive_training.training_types import (
    TrainingType,
    normalize_training_type,
    training_type_field,
)
from interactive_training.utils import file_utils

logger = logging.getLogger(__name__)


def _parse_checkpoint_path(checkpoint_path: str) -> tuple[str, str]:
    """Parse a checkpoint path into ``(session_id, checkpoint_name)``.

    Accepts ``<session_id>/<name>`` URIs (as written to
    checkpoints.jsonl) or plain ``<session_id>/<checkpoint_name>`` paths.
    Raises ``ValueError`` on any other shape.
    """
    if "://" in checkpoint_path:
        stripped = checkpoint_path.split("://", 1)[1]
        parts = [p for p in stripped.split("/") if p]
        if len(parts) >= 2:
            session = parts[0]
            if not session.startswith("session_"):
                session = f"session_{session.removeprefix('model_')}"
            return session, parts[-1]

    parts = checkpoint_path.split("/", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        session = parts[0]
        if not session.startswith("session_"):
            session = f"session_{session.removeprefix('model_')}"
        return session, parts[1]

    raise ValueError(
        "load_checkpoint_path must be '<session_id>/<name>' or "
        f"'<session_id>/<checkpoint_name>', got: {checkpoint_path!r}"
    )


@chz.chz
class CLIConfig:
    """Command-line config for on-policy distillation via the Azure SDK."""

    # ── Student model ────────────────────────────────────────────────────────
    model_name: str = model_info.DEFAULT_MODEL_NAME
    tokenizer_name: str | None = None
    renderer_name: str | None = None
    lora_rank: int = 16
    load_checkpoint_path: str | None = None

    # ── Teacher (reference) model ────────────────────────────────────────────
    # Defaults to the student's base model — for a smoke test the teacher does
    # not need to be good at any task.
    teacher_model: str | None = None
    teacher_checkpoint: str | None = None

    # ── Dataset (prompt-only Tulu3) ──────────────────────────────────────────
    max_train_examples: int = 64
    max_prompt_chars: int | None = 2048
    streaming: bool = True
    seed: int = 0

    # ── Training hyperparameters ─────────────────────────────────────────────
    group_size: int = 4
    groups_per_batch: int = 8
    max_concurrent_groups: int | None = None
    learning_rate: float = 1e-5
    max_tokens: int = 512
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    num_substeps: int = 1
    loss_fn: LossFnType = "importance_sampling"

    # ── Logging / eval / checkpointing ───────────────────────────────────────
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 999999
    save_every: int = 999999
    max_steps: int | None = None

    # ── Azure SDK ────────────────────────────────────────────────────────────
    project_endpoint: str
    verbose_http: bool = False
    training_type: TrainingType | None = training_type_field()
    user_metadata: dict[str, Any] | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig) -> None:
    cli_config_dump = {
        key: getattr(cli_config, key, None) for key in CLIConfig.__annotations__.keys()
    }
    print(
        "[distillation] cli_config:\n"
        + json.dumps(cli_config_dump, indent=2, sort_keys=True, default=str)
    )

    tokenizer_name = cli_config.tokenizer_name or cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    teacher_model = cli_config.teacher_model or cli_config.model_name

    model_name_safe = cli_config.model_name.replace("/", "-")
    run_name = (
        f"distill-tulu3-azure-sdk-{model_name_safe}"
        f"-{cli_config.lora_rank}rank"
        f"-{cli_config.learning_rate}lr"
        f"-{cli_config.group_size}group"
        f"-{cli_config.groups_per_batch}batch"
        f"-kl{cli_config.kl_penalty_coef}"
        f"-seed{cli_config.seed}"
        f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    log_path = (
        cli_config.log_path
        if cli_config.log_path is not None
        else str(file_utils.default_logs_root() / "distillation" / run_name)
    )
    wandb_name = cli_config.wandb_name or run_name

    dataset_builder = TuluPromptDatasetBuilder(
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=tokenizer_name,
        renderer_name=renderer_name,
        max_train_examples=cli_config.max_train_examples,
        max_prompt_chars=cli_config.max_prompt_chars,
        seed=cli_config.seed,
        streaming=cli_config.streaming,
    )

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=tokenizer_name,
        tokenizer_name=tokenizer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        kl_reference_config=KLReferenceConfig(
            base_model=teacher_model,
            load_checkpoint_path=cli_config.teacher_checkpoint,
        ),
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_steps=cli_config.max_steps,
        sampling_seed=cli_config.seed,
        max_concurrent_groups=cli_config.max_concurrent_groups,
    )

    print(f"[distillation] resolved training config: {config!r}")

    cli_utils.check_log_dir(
        log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    os.makedirs(log_path, exist_ok=True)
    run_meta = {
        "azure": {
            "session_id": None,
            "reference_session_id": None,
            "project_endpoint": cli_config.project_endpoint,
        },
        "model": {"model_name": cli_config.model_name, "teacher_model": teacher_model},
    }

    def write_run_meta() -> None:
        path = os.path.join(log_path, "run_meta.json")
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            file_utils.set_private_file_permissions(descriptor, path)
            json.dump(run_meta, output, indent=2)

    write_run_meta()

    def record_reference_session(session_id: str) -> None:
        run_meta["azure"]["reference_session_id"] = session_id
        write_run_meta()

    # ── Azure SDK: auth + student session creation ───────────────────────────
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

    # Honor an explicit student checkpoint: create the student session from it
    # (otherwise create_session starts from the base model and the checkpoint is
    # silently ignored). rl_azure.main runs on the already-created session, so
    # this must be applied here at session creation.
    student_from_checkpoint: "FromCheckpoint | None" = None
    if cli_config.load_checkpoint_path:
        src_session, ckpt_id = _parse_checkpoint_path(cli_config.load_checkpoint_path)
        student_from_checkpoint = FromCheckpoint(
            source_session_id=src_session, checkpoint_id=ckpt_id
        )

    print(
        f"Creating student session: base_model={cli_config.model_name}  "
        f"lora_rank={cli_config.lora_rank}  endpoint={cli_config.project_endpoint}  "
        f"load_checkpoint_path={cli_config.load_checkpoint_path!r}"
    )
    try:
        session_id = await client.create_session(
            base_model=cli_config.model_name,
            lora_config=LoRAConfig(rank=cli_config.lora_rank),
            type="training",
            from_checkpoint=student_from_checkpoint,
            timeout_sec=600.0,
            user_metadata=cli_config.user_metadata,
            training_type=normalize_training_type(cli_config.training_type),
        )
        print(f"Student session ready: session_id={session_id}")
        run_meta["azure"]["session_id"] = session_id
        write_run_meta()
        print(
            f"KL reference (teacher): base_model={teacher_model}  "
            f"checkpoint={cli_config.teacher_checkpoint}  "
            f"(created inside rl_azure.main when kl_penalty_coef > 0)"
        )

        await rl_azure.main(
            config, client, session_id, training_type=cli_config.training_type,
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
