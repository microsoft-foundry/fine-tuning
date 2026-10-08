"""On-policy distillation building blocks for the interactive-training cookbook.

Single-turn on-policy distillation: the student samples completions for a
prompt, the teacher (reference) model scores those completions' log-probs, and
the only training signal is the KL penalty against the teacher (no correctness
or format rewards). This reuses the existing RL training loop in
``interactive_training.rl.train`` — the teacher is wired in as the ``kl_reference``.

There is deliberately no environment logic here (no grading, sandbox, or
tools): the ``PromptOnlyEnv`` returns reward ``0.0`` and the KL penalty does
all the work.
"""

from interactive_training.distillation.datasets import (
    PromptOnlyDataset,
    PromptOnlyEnv,
    TeacherConfig,
    TuluPromptDatasetBuilder,
)

__all__ = [
    "PromptOnlyDataset",
    "PromptOnlyEnv",
    "TeacherConfig",
    "TuluPromptDatasetBuilder",
]
