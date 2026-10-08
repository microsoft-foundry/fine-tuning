# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [3.0.0] - 2026-10-07

Interactive training and managed fine-tuning refresh ([#42](https://github.com/microsoft-foundry/fine-tuning/pull/42)).

### Added
- Interactive training cookbook, training recipes, SDK guidance, and supporting documentation.
- Refresh shared documentation, tests, and repository guidance for both training tracks.
- Explicit access-approval notices and the preview sign-up link for interactive training.

### Changed
- Reorganize existing fine-tuning examples, datasets, and notebooks into a `managed_fine_tuning` folder

### Removed
- Legacy standalone fine-tuning agent skill and its auto-discovery symlinks; maintained repository guides and recipes now define the supported workflows.

## [2.1.0] - 2026-04-21

### Added
- `monitor_training.py` — poll a running job until completion with real-time event streaming
- `calibrate_grader.py` — calibrate RFT grader pass_threshold on base model outputs
- `cleanup.py` — list and delete old files, deployments, and pending jobs to reclaim quota
- `workflows/quickstart.md` — 6-step guide from zero to a deployed fine-tuned model
- `references/grader-design.md` — comprehensive RFT grader design guide (type selection, partial credit, threshold calibration)
- RFT-specific metrics section in `references/training-curve-analysis.md` (reward curves, token growth, parse errors, checkpoint selection)
- LLM augmentation approach in `workflows/dataset-creation.md`
- Agent discovery symlinks (`.github/skills/`, `.claude/skills/`, `.agents/skills/`) for auto-discovery by Copilot, Claude, and Codex
- PEP 723 inline script metadata on all scripts for `uv run` support
- `HelpOnErrorParser` for better CLI error messages across all scripts
- `DefaultAzureCredential` fallback in `common.py` for keyless authentication
- AI Agent Skills section in README with usage instructions for all supported agents

### Changed
- Updated hyperparameter recommendations in `references/agentic-rft.md` (LR=1.0, compute_multiplier=1.5)
- Restructured `workflows/dataset-creation.md` into three equal approaches (manual, LLM augmentation, synthetic generation)
- Updated `references/evaluation-methodology.md` to include token cost alongside accuracy
- Generalized experiment-specific numbers across all docs

### Fixed
- `submit_training.py`, `deploy_model.py`: Added missing `requests` to PEP 723 dependencies
- `submit_training.py`: Fixed file handle leak when reading grader files
- `evaluate_model.py`: Fixed `StopIteration` crash on malformed test data
- `validate/validate_rft.py`: Fixed broken newline escape detection logic
- `convert_dataset.py`: Fixed DPO generation to include system messages for base model

## [2.0.0] - 2026-04-16

### Added
- **Azure AI Fine-Tuning coding agent skill** (`Skills/`) — comprehensive agent skill for SFT, DPO, and RFT workflows on Azure AI Foundry
  - `SKILL.md` with full lifecycle guidance, platform gotchas, and rules
  - 8 reusable Python scripts (submit, check, deploy, evaluate, convert, generate, score, common)
  - Data validators for SFT, DPO, and RFT formats with cost estimates
  - 11 reference docs covering hyperparameters, dataset formats, deployment, evaluation, agentic RFT, reward hacking prevention, vision FT, cost management, and more
  - 5 guided workflows (full pipeline, dataset creation, iterative training, diagnosis, experiment review)
  - Sample data for SFT, DPO, and RFT formats
- RFT best practice guidance (PR #24, contributed by Blanca Li)

## [1.1.0] - 2026-02-02

### Added
- Enhanced main README with navigation table, quickstart section, and demo links
- Getting Started guide (`GETTING_STARTED.md`) with step-by-step setup instructions
- Schema documentation (`SCHEMA.md`) for all managed fine-tuning datasets (SFT, DPO, RFT)
- Troubleshooting sections to demos that were missing them
- Missing `.env.template` and `requirements.txt` for RFT_Countdown demo
- Issue templates for bug reports and feature requests
- This CHANGELOG file

### Changed
- Improved discoverability with direct links from README to demos and datasets
- Standardized troubleshooting format across demo READMEs

### Fixed
- Fixed placeholder URLs in CONTRIBUTING.md

## [1.0.0] - 2025-01-15

### Added
- Initial release with 12 end-to-end demos
- Sample datasets for SFT, DPO, and RFT techniques
- Demos covering text, image, video, and multimodal fine-tuning
- Support for Azure OpenAI and OSS models
