# Microsoft Foundry Fine-Tuning

[Microsoft Foundry](https://ai.azure.com/) fine-tuning adapts pretrained models to your data and tasks, creating domain-specific responses. Compared to prompt or agent optimization, fine-tuning changes the actual model's behavior to improve quality or reduce costs.  This repo provides examples, datasets, and guidance for two complementary experiences offered by Microsoft Foundry: **managed fine-tuning** and **interactive training (preview)**.

> [!IMPORTANT]
> Interactive training requires explicit access approval. Request access through the [preview sign-up form](https://aka.ms/foundry-interactive-training-signup) and wait for approval before creating a training session.

**🌱 New to fine-tuning? Start with managed fine-tuning.** Interactive training is for advanced workflows that need custom training-loop behavior.

| | Managed Fine-Tuning | Interactive Training (preview) |
|---|---|---|
| **How it works** | Submit data and settings; Foundry runs the training job. | Write a Python loop; Foundry executes training and sampling operations. |
| **Best for** | Beginners and AI engineers who prefer predefined workflows over lower-level training operations. | Experienced ML practitioners building custom training workflows. |
| **Your control** | Data, supported hyperparameters, and RFT graders. | Losses, rewards, rollouts, gradient accumulation, updates, and checkpoints. |
| **Training methods** | Model-specific SFT, DPO, and RFT; distillation through teacher-generated SFT data. | Recipes for SFT, reinforcement learning, preference learning, distillation, and custom losses. |
| **Example use cases** | Distill a larger model, learn from support conversations, or improve responses with a grader. | Collect tool-use rollouts, apply custom rewards, or change training-loop update and evaluation behavior. |

Both approaches use Foundry-managed training infrastructure; neither requires you to provision the training GPUs.

**Need a specific model?** [Model support](#models-and-supported-capabilities) may determine your approach. [Learn when to fine-tune →](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/fine-tuning-considerations)

## 🚀 Get started

Open the folder for your chosen experience and start with its guide for setup and examples:

- **🌱 Managed fine-tuning guide →** [managed_fine_tuning/README.md](managed_fine_tuning/README.md)
- **🧪 Interactive training guide →** [interactive_training/README.md](interactive_training/README.md)

## Models and supported capabilities

Selected models are listed below. Availability also depends on region, access, and quota.

See the [model and region registry](interactive_training/docs/supported_models.md) for interactive model identifiers and regions. Managed methods are model-specific; check [Foundry availability](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning#supported-models) and the [managed model/notebook index](managed_fine_tuning/MODEL_AND_NOTEBOOK_INDEX.md) before starting a job.

| Model | Managed Fine-Tuning | Interactive Training (preview) |
|---|---|:---:|
| `Qwen3.8-27B` **(preview)** | ✅ (SFT, RFT) | ✅ |
| `Qwen3.6-35B-A3B` **(preview)** | ✅ (SFT, RFT) | ✅ |
| `gpt-oss-120b` **(preview)** | ✅ (SFT, RFT) | ✅ |
| `Muse-Glimmer-30B` **(preview)** | ✅ (SFT, RFT) | ✅ |
| `gpt-4o-mini` (2024-07-18) | ✅ (SFT) | ❌ |
| `gpt-4o` (2024-08-06) | ✅ (SFT, DPO) | ❌ |
| `gpt-4.1` (2025-04-14) | ✅ (SFT, DPO) | ❌ |
| `gpt-4.1-mini` (2025-04-14) | ✅ (SFT, DPO) | ❌ |
| `gpt-4.1-nano` (2025-04-14) | ✅ (SFT, DPO) | ❌ |
| `o4-mini` (2025-04-16) | ✅ (RFT) | ❌ |
| `gpt-5` (2025-08-07) (invitation-only) | ✅ (RFT) | ❌ |
| `Ministral-3B` (2411) | ✅ (SFT) | ❌ |
| `Qwen3-32B` | ✅ (SFT) | ❌ |
| `Llama-3.3-70B-Instruct` | ✅ (SFT) | ❌ |
| `gpt-oss-20b` | ✅ (SFT) | ❌ |

## 📂 Repository guide

| Resource | What you'll find |
|---|---|
| [Managed fine-tuning](managed_fine_tuning/README.md) | Managed training guide. |
| [Managed fine-tuning learning path](managed_fine_tuning/LEARNING_PATH.md) | Six-stage curriculum and 14 canonical notebooks. |
| [Managed fine-tuning task index](managed_fine_tuning/TASK_INDEX.md) | Find a notebook by task; each demo owns its data and setup template. |
| [Interactive training](interactive_training/README.md) | Interactive cookbook, recipes, and SDK guidance. |
| [Interactive training docs](interactive_training/docs/README.md) | Setup, training concepts, checkpoints, and troubleshooting. |
| [Interactive training recipes](interactive_training/interactive_training/recipes/) | Runnable SFT, reinforcement learning, preference, and distillation recipes. |
| [Paid smoke operations](PAID_SMOKE_TESTING.md) | Opt-in workflow setup, execution bounds, private evidence, and recovery. |

**For coding agents:** start with [AGENTS.md](AGENTS.md) for source precedence, offline preflight, paid-run warnings, scope safeguards, and secret handling. The two training paths are not interchangeable.

> [!NOTE]
> **Before production:** evaluate quality and safety on held-out data and review model/data licenses. Samples are for experimentation. Training, tools, and serving may incur separate charges; clean up unused resources.

Managed notebooks are offline-validated, not evidence of a successful cloud run. Most stop at service training metrics; only Retail adds comparisons using separately provisioned deployments. Training completion alone does not establish application quality or create a serving endpoint.

## 🤝 Contributing

Examples, datasets, and documentation improvements are welcome. Read the [contribution guidelines](CONTRIBUTING.md) and [Code of Conduct](code_of_conduct.md). The CLA bot will guide you if a Contributor License Agreement is required.

[MIT License](LICENSE); models, datasets, and dependencies may have separate terms.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is subject to those third parties' policies.
