# Interactive Training Cookbook Docs

Follow a complete learning journey, or use the reference pages for an existing run. All commands assume the **cookbook directory** (`fine-tuning/interactive_training`), not the parent repository root. The outer cookbook and inner Python package now both use `interactive_training`; module commands use `python -m interactive_training...`.

## Choose your next step

| Question | Guide | Completion evidence |
|---|---|---|
| Is this the right approach for my task? | [Concepts](./concepts.md) and [recipe chooser](./recipes.md#choose-a-path) | You can identify SFT, RL, preference learning, or distillation and its prerequisites |
| How do I get one run working? | [Quickstart](./quickstart.md) | Imports/help pass, a bounded remote run writes metrics/checkpoints, cleanup is checked |
| Does my project have the right access? | [Authentication](./auth.md), [models and regions](./supported_models.md) | Correct project endpoint, credential, model eligibility, and capacity |
| How do I use my own data and rewards? | [Custom data](./custom-data.md) | JSONL renders locally with valid masks; the programmatic SFT launch warns before paid execution and reuses validated data; RL grader checks pass |
| Did training help, and how do I use the result? | [Evaluation and inference](./evaluation-and-inference.md) | Comparable held-out measurements and sampling from a saved checkpoint |
| Where are my artifacts? | [Storage](./storage.md) and [dashboard](./dashboard.md) | Config, metrics, ledger, and sensitive artifacts can be located and interpreted |
| How do I recover or free resources? | [Recovery](./training.md#resuming-after-an-interruption), [continual training](./continual-fine-tuning.md), [session management](./session-management.md) | The intended checkpoint/cursor is loaded; owned sessions are inspected and explicitly closed |
| What can a coding agent safely run? | [Repository execution contract](https://github.com/microsoft-foundry/fine-tuning/blob/main/AGENTS.md) and [contributing](./contributing.md) | Offline checks are separated from downloads and paid/destructive operations |

## CLI conventions

**Training launchers using `chz`:** arguments are `key=value`; no leading `--`, no spaces around `=`, and booleans are `true`/`false`. Discover the selected launcher's fields before copying knobs from another recipe:

```bash
python -m interactive_training.recipes.math_rl.train_azure --help
python -m interactive_training.recipes.tulu3_sft.train_azure --help
```

**Different parsers:** the dashboard uses flags such as `--root`; visual-spatial `sample_azure` uses flags such as `--project-endpoint`. Do not apply the chz rule to these commands:

```bash
python dashboard_server.py --help
python -m interactive_training.recipes.visual_spatial.sample_azure --help
```

Help displays the actual CLI contract without creating a session. A field on a programmatic `Config` is **not necessarily** a launcher argument. In PowerShell, use its syntax or a single-line command; Bash `\` continuation does not work there.

## Reference map

### Training and extension

- [Training](./training.md): RL/SFT loops, tunable parameters, budgets, and cadence.
- [Loss functions](./loss_functions.md): wire inputs, built-in losses, custom differentiable losses.
- [SDK reference](./sdk-reference.md): raw SDK versus cookbook adapters and their await patterns.
- [Custom data](./custom-data.md): chat JSONL and prompt/grader adaptation using existing builders.
- [Continual fine-tuning](./continual-fine-tuning.md): fresh run from saved training state, versus cursor-preserving recovery.
- [Contributing](./contributing.md): recipe structure and documentation/test conventions.

### Evaluation and operations

- [Evaluation and inference](./evaluation-and-inference.md): baseline/final comparisons, saved checkpoint sampling, and deployment boundaries.
- [Storage](./storage.md): local files versus remote state, artifact timing, and retention caveats.
- [Dashboard](./dashboard.md): local visualization and private port forwarding.
- [Session management](./session-management.md): discovery, checkpoint metadata, and explicitly scoped cleanup.
- [Troubleshooting](./troubleshooting.md): installation, authentication, output truncation, restart, and cleanup failures.

### Examples and availability

- [Recipes](./recipes.md): paths, dependencies, and default model mapping.
- [Supported models](./supported_models.md): exact identifiers, current eligibility qualifications, and project/region admission limits.
- [Benchmarks](./benchmarks.md): historical, model-specific measurements—not expected results for a new default or a smoke run.
