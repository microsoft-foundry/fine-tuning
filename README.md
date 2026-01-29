# Fine-Tuning in AI Foundry

This repository contains **12 end-to-end demos** and **sample datasets** for fine-tuning models on [Azure AI Foundry](http://ai.azure.com/). Use this repo to explore practical fine-tuning workflows and access ready-to-use data for your own projects.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Demos](#-demos)
- [Sample Datasets](#-sample-datasets)
- [Prerequisites](#-prerequisites)
- [Contributing](#contributing)

---

## 🚀 Quick Start

**New to fine-tuning?** Start here:

1. **[SFT_CNN_DailyMail](Demos/SFT_CNN_DailyMail/)** - Best first demo! Learn Supervised Fine-Tuning with news summarization
2. **[Sample_Datasets](Sample_Datasets/)** - Understand data formats for SFT, DPO, and RFT

**Ready for advanced techniques?**
- **[DPO_Intel_Orca](Demos/DPO_Intel_Orca/)** - Direct Preference Optimization
- **[RFT_Countdown](Demos/RFT_Countdown/)** - Reinforcement Fine-Tuning

---

## 🎯 Demos

Explore end-to-end fine-tuning experiences in the **[Demos](Demos/)** folder:

| Demo | Technique | Use Case | Difficulty |
|------|-----------|----------|------------|
| [SFT_CNN_DailyMail](Demos/SFT_CNN_DailyMail/) | SFT | News summarization | ⭐ Beginner |
| [SFT_PubMed_Summarization](Demos/SFT_PubMed_Summarization/) | SFT | Medical paper summarization | ⭐ Beginner |
| [DPO_Intel_Orca](Demos/DPO_Intel_Orca/) | DPO | Preference optimization | ⭐⭐ Intermediate |
| [RFT_Countdown](Demos/RFT_Countdown/) | RFT | Math puzzle solving | ⭐⭐ Intermediate |
| [DistillingSarcasm](Demos/DistillingSarcasm/) | Distillation | Knowledge transfer | ⭐⭐ Intermediate |
| [Image_Breed_Classification_FT](Demos/Image_Breed_Classification_FT/) | Vision SFT | Dog breed classification | ⭐⭐ Intermediate |
| [Image_FT_Chart_Analysis](Demos/Image_FT_Chart_Analysis/) | Vision SFT | Chart understanding | ⭐⭐ Intermediate |
| [Video_FT_Action_Recognition](Demos/Video_FT_Action_Recognition/) | Vision SFT | Video action detection | ⭐⭐⭐ Advanced |
| [ZavaRetailAgent](Demos/ZavaRetailAgent/) | SFT + RFT | Retail customer service agent | ⭐⭐⭐ Advanced |
| [Agentic_RFT_PrivatePreview](Demos/Agentic_RFT_PrivatePreview/) | RFT | Agentic workflows with tools | ⭐⭐⭐ Advanced |
| [Evaluation](Demos/Evaluation/) | Evaluation | Multimodal model evaluation | ⭐⭐ Intermediate |

👉 See **[Demos/README.md](Demos/README.md)** for detailed descriptions of each demo.

---

## 📊 Sample Datasets

Ready-to-use datasets for testing fine-tuning techniques in the **[Sample_Datasets](Sample_Datasets/)** folder:

| Technique | Dataset | Description |
|-----------|---------|-------------|
| **SFT** | [Text-GSM8K](Sample_Datasets/Supervised_Fine_Tuning/Text-GSM8K/) | Grade school math problems |
| **SFT** | [Multimodal-chartqa](Sample_Datasets/Supervised_Fine_Tuning/Multimodal-chartqa/) | Chart interpretation |
| **SFT** | [Tool-Calling](Sample_Datasets/Supervised_Fine_Tuning/Tool-Calling/) | Function calling patterns |
| **DPO** | [orca_dpo_pairs](Sample_Datasets/Direct_Preference_Optimization/orca_dpo_pairs/) | Preference alignment |
| **RFT** | [ClauseMatching](Sample_Datasets/Reinforcement_Fine_Tuning/ClauseMatching/) | Legal contract analysis |
| **RFT** | [MedMCQ](Sample_Datasets/Reinforcement_Fine_Tuning/MedMCQ/) | Medical Q&A |

👉 See **[Sample_Datasets/README.md](Sample_Datasets/README.md)** for data format details and when to use each technique.

> ⚠️ **Note**: These datasets are for **learning and experimentation only**—not for production use. Training jobs may incur costs on your Azure subscription.

---

## ✅ Prerequisites

Before running any demo, ensure you have:

- **Azure subscription** with access to [Azure AI Foundry](http://ai.azure.com/)
- **Python 3.9+** installed
- **Jupyter Notebook** or VS Code with Jupyter extension
- Required **Azure role assignments** (see individual demo READMEs)

Each demo includes a `requirements.txt` and `.env.template` for setup.

👉 **New here?** See the **[Getting Started Guide](GETTING_STARTED.md)** for step-by-step setup instructions.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues and pull requests.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
