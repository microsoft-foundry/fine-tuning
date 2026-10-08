"""Prompt-only datasets for on-policy distillation.

The only supervision comes from the KL penalty against the teacher model, so the
environment provides no correctness or format reward. For a fast smoke test we
reuse interactive-training's existing Tulu3 source (``allenai/tulu-3-sft-mixture``, already wired
into ``recipes/tulu3_sft``) but stream only a handful of prompts and keep just
the first user turn.
"""

import logging
import math
from functools import partial
from typing import Sequence

import chz
from azure.ai.finetuningsessions.models import ModelInput

from interactive_training import renderers
from interactive_training.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from interactive_training.rl.types import (
    Action,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from interactive_training.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

_TULU3_DATASET = "allenai/tulu-3-sft-mixture"


@chz.chz
class TeacherConfig:
    """Configuration for the teacher (reference) model used for the KL signal.

    For a smoke test, set ``base_model`` equal to the student's base model — the
    teacher does not need to be good at any task to exercise the code path.
    """

    base_model: str
    load_checkpoint_path: str | None = None


class PromptOnlyEnv(ProblemEnv):
    """Zero-reward environment: the only supervision is KL against the teacher.

    Subclasses ``ProblemEnv`` to reuse its ``initial_observation`` (renders the
    prompt into a generation prompt), but overrides ``step`` to always return a
    reward of ``0.0`` and disables format checking.
    """

    def __init__(
        self,
        prompt: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        # format_coef=0.0: no format shaping, KL is the only signal.
        super().__init__(renderer, convo_prefix, format_coef=0.0)
        self.prompt = prompt

    def get_question(self) -> str:
        return self.prompt

    def check_answer(self, sample_str: str) -> bool:
        return False

    def check_format(self, sample_str: str) -> bool:
        return True

    def get_reference_answer(self) -> str:
        return ""

    async def step(self, action: Action) -> StepResult:
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=ModelInput(chunks=[]),
            next_stop_condition=self.stop_condition,
            metrics={},
        )


class PromptOnlyDataset(RLDataset):
    """Serves batches of prompt-only, zero-reward env groups."""

    def __init__(
        self,
        prompts: list[str],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        dataset_name: str = "tulu3",
    ):
        if not prompts:
            raise ValueError("PromptOnlyDataset requires at least one prompt")
        self.prompts = prompts
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.dataset_name = dataset_name

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.prompts))
        assert batch_start < batch_end, "Incorrect batch size / index out of range"
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    PromptOnlyEnv,
                    prompt,
                    renderer=self.renderer,
                    convo_prefix=self.convo_prefix,
                ),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            for prompt in self.prompts[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.prompts) / self.batch_size)


def load_tulu3_prompts(
    n: int,
    *,
    seed: int = 0,
    streaming: bool = True,
    max_prompt_chars: int | None = 2048,
) -> list[str]:
    """Load ``n`` prompts (first user turn) from the Tulu3 SFT mixture.

    Uses the same HF dataset as ``recipes/tulu3_sft``. With ``streaming=True``
    only a handful of rows are pulled (no multi-GB download) — ideal for a
    smoke test.
    """
    import datasets

    ds = datasets.load_dataset(_TULU3_DATASET, split="train", streaming=streaming)
    if streaming:
        ds = ds.shuffle(seed=seed, buffer_size=max(n * 8, 1000))
    else:
        ds = ds.shuffle(seed=seed)

    prompts: list[str] = []
    for row in ds:
        messages = row.get("messages") or []
        first_user = next(
            (m["content"] for m in messages if m.get("role") == "user" and m.get("content")),
            None,
        )
        if not first_user:
            continue
        if max_prompt_chars is not None:
            first_user = first_user[:max_prompt_chars]
        prompts.append(first_user)
        if len(prompts) >= n:
            break

    if not prompts:
        raise RuntimeError(f"Could not load any prompts from {_TULU3_DATASET}")
    logger.info("Loaded %d Tulu3 prompts (streaming=%s)", len(prompts), streaming)
    return prompts


@chz.chz
class TuluPromptDatasetBuilder(RLDatasetBuilder):
    """Builds a prompt-only Tulu3 dataset for on-policy distillation."""

    groups_per_batch: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    max_train_examples: int = 64
    max_prompt_chars: int | None = 2048
    seed: int = 0
    streaming: bool = True
    convo_prefix: list[renderers.Message] | None = None

    async def __call__(self) -> tuple[PromptOnlyDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        prompts = load_tulu3_prompts(
            self.max_train_examples,
            seed=self.seed,
            streaming=self.streaming,
            max_prompt_chars=self.max_prompt_chars,
        )
        train_dataset = PromptOnlyDataset(
            prompts=prompts,
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=self.convo_prefix,
            dataset_name="tulu3",
        )
        return train_dataset, None
