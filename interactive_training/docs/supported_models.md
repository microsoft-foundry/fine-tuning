# Supported Models

## Models

### Available today

| Model | Identifier (`model_name` / `tokenizer_name`) | Availability |
|---|---|:---:|
| qwen38-27b | `Qwen/Qwen3.8-27B` | Available |
| qwen36-35b-a3b | `Qwen/Qwen3.6-35B-A3B` | Available |
| gpt-oss-120b | `openai/gpt-oss-120b` | Available |
| muse-glimmer-30b | `meta-models/Muse-Glimmer-30B` | Available |

Model availability also depends on model and region capacity;
these status labels do not guarantee quota.

Pass the **model identifier** as both `model_name` and `tokenizer_name` to the
recipes (e.g. `model_name="Qwen/Qwen3.8-27B" tokenizer_name="Qwen/Qwen3.8-27B"`).
Shared text recipes default to `Qwen/Qwen3.8-27B`; omitting `tokenizer_name`
uses the selected model's tokenizer. The general Qwen3.8 renderer default is
`qwen3_8_low_reasoning`; explicit renderer overrides and specialized non-thinking
preferences are preserved. Renderer support in code does not imply service
availability beyond the models listed below.

Model inferencing is offered through the **Fireworks private preview**. For more information and to request access, see
[aka.ms/fireworks-consent](https://aka.ms/fireworks-consent).

The client needs only public tokenizer files, not local model weights or a
private tokenizer path.

**Keep identifiers exact.** Model name, tokenizer, renderer, and LoRA settings
must remain compatible with a saved checkpoint. A new default model does not
make an older checkpoint portable to another base model. See
[checkpoint reuse](./evaluation-and-inference.md#use-a-training-checkpoint-after-the-original-session-closes).

### Legacy models

These identifiers appear in historical recipes and benchmarks but are no longer supported
for new interactive training.

| Model | Identifier (`model_name` / `tokenizer_name`) | Availability |
|---|---|:---:|
| qwen3-32b | `Qwen/Qwen3-32B` | ⚠️ Legacy |
| gpt-oss-20b | `openai/gpt-oss-20b` | ⚠️ Legacy |
| qwen35-4b | `Qwen/Qwen3.5-4B` | ⚠️ Legacy |
| qwen35-9b | `Qwen/Qwen3.5-9B` | ⚠️ Legacy |


### Vision fine-tuning

Image-bearing SFT or RL data requires both:

1. A vision-language model architecture.
2. Image fine-tuning support for that model in the service.

A vision renderer does not add image support to a text-only model. The cookbook
checks this before launching the `visual_spatial` RL recipe. Confirm that image
fine-tuning is enabled for your model and project before submitting image-bearing
data. The model-agnostic
[visual-spatial recipe](../interactive_training/recipes/visual_spatial/) selects the
recommended renderer for any supported vision-language model.

## Available regions

The following regions support Interactive Training:

| Deployment group | Region | Region ID |
|---|---|---|
| US | East US | `eastus` |
| US | East US 2 | `eastus2` |
| US | North Central US (NCUS) | `northcentralus` |
| US | West US 3 | `westus3` |
| US | Canada East | `canadaeast` |
| Europe | Sweden Central | `swedencentral` |
| Europe | Switzerland North | `switzerlandnorth` |
| APAC | Australia East | `australiaeast` |
| APAC | Korea Central | `koreacentral` |
| UAE | UAE North | `uaenorth` |

Your Azure AI Foundry resource must live in one of these regions — see
[auth.md](./auth.md#prerequisite--an-azure-ai-foundry-resource).
