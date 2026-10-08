# Contributing

The cookbook is intentionally small. New content lands as either a **recipe** (a runnable end-to-end example) or a **docs page** (under `docs/`).

## Adding a recipe

A recipe is a directory under `interactive_training/recipes/<name>/` with:

```
interactive_training/recipes/<name>/
├── README.md              # launch command, expected metrics, example trajectory
├── train_azure.py         # CLI entry point (the Azure SDK variant)
├── <task>_env.py          # prompt building + step()
└── <task>_grading.py      # reward function
```

Public recipes use the pinned Azure SDK and the documented project endpoint.
Do not require a separate private runtime, checkout, or helper script to set up
the recipe.

Conventions:

- **Start from [custom-data.md](./custom-data.md)** for existing SFT/RL data contracts. Wire a task builder/grader into its own launcher; do not fork the shared algorithm or invent CLI fields when a programmatic interface is enough.
- **Document the complete lifecycle:** prerequisites/security and data downloads, a bounded smoke command using real CLI fields, expected metric/artifact names, a held-out [evaluation plan](./evaluation-and-inference.md), checkpoint use/recovery, and session cleanup. Separate local checks from paid service calls and from execution of generated code or external tools.
- **Entry point uses `chz`** for argument parsing. Mirror the style in `recipes/math_rl/train_azure.py`.
- **Use `model_info.DEFAULT_MODEL_NAME` for general text recipes.** It is currently
  `Qwen/Qwen3.8-27B`. Let tokenizer/renderer selection follow the model; do not
  hardcode a renderer from a different family. Automatic Qwen3.8 selection uses
  `qwen3_8_low_reasoning`; the explicit `qwen3_8` name still means `xhigh`.
  Vision recipes must require or choose a vision-language model instead.
- **Expose `training_type: TrainingType | None = training_type_field()`** on
  training entrypoints, importing both names from `interactive_training.training_types`.
  The shared `TrainingType` literal currently accepts only `GlobalStandard`
  (case-insensitive); `DatazoneStandard` and `DeveloperTier` are rejected as
  unavailable. Leave `None` omitted when creating a session so the service
  selects the tier. Use
  `normalize_training_type` at programmatic boundaries. RL recipes must also
  forward the optional tier to `rl.train_azure.main` so teacher/KL-reference
  sessions use it too.
- **Log to `<logs-root>/<recipe>/<run_name>/`** using the helpers in `interactive_training/utils/`. The default logs root is resolved via `default_logs_root()` — don't hard-code paths.
- **Write `run_meta.json`** at launch so the dashboard's Run Overview card has something to show.
- **Save checkpoints to `checkpoints.jsonl`** via `interactive_training.checkpoint_utils.save_checkpoint_async` inside async launchers (`save_checkpoint` is the synchronous wrapper). Await all requested saves; preserve the complete resume cursor and returned paths. Only training state is resumable; distinguish persistent sampler saves from ephemeral refreshes ([SDK boundary](./sdk-reference.md#persistent-versus-ephemeral-checkpoints)).
- **Add the recipe to [recipes.md](./recipes.md)** with a one-line description.

## Adding a docs page

Each `docs/*.md` page should:

- Open with one sentence saying what the page is about and (if relevant) what it's *not* about (with a link to the right page instead).
- Use GitHub-style alerts for callouts: `> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`. They render natively on GitHub and degrade to plain blockquotes elsewhere.
- Use relative links between docs pages and into source (`../interactive_training/...`).
- Cross-link aggressively — readers should never hit a dead end.

Add the page to the TOC in [docs/README.md](./README.md) and to the nav table in the top-level [README.md](../README.md) if it's a top-level concept.

## Validate before publishing

From the cookbook root, use an isolated environment and the published SDK
from the [installation guide](../README.md#install):

```bash
python -m pip install -e '.[test]' --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip check
python -m interactive_training.recipes.math_rl.train_azure --help
python -m pytest -q tests/test_public_documentation.py --tb=short
python -m pytest -q --tb=short
```

Tests cover CLI defaults, documentation links, notebook syntax/output hygiene,
SDK example contracts, and recipe/runtime regressions. Tokenizer parity tests
may download public tokenizer files; some optional model fixtures are skipped
when unavailable. The automated suite must not launch paid Azure training.
Live examples and deployment checks require a separate, explicitly configured
environment. Validate the cookbook against its pinned published SDK.

For a documentation-only change, keep source, dependencies, tests, CI, and notebook code unchanged. Review the diff for scope, validate relative links **and anchors** (the existing link test checks paths), compile fenced Python examples, and bind command arguments to the actual `CLIConfig` without calling the launcher. Mock remote calls for helper contract/cleanup tests; never run a paid example to check syntax. Use the [repository execution contract](https://github.com/johnwu0604/fine-tuning/blob/main/AGENTS.md) for coding-agent safety boundaries. Report exactly what was tested and what still needs authorized live validation; neither mocked helpers nor an existing environment prove a fresh PyPI install or quality improvement. Keep temporary score reports, audits, and validation harnesses outside the repository; the public guides document supported workflows, not an agent's private analysis.

Clear notebook outputs before sharing: they can contain session IDs, prompts,
dataset text, and results from an older configuration. Do not relabel historical
benchmark results when changing defaults. Keep examples and tests suitable for
the public preview.
