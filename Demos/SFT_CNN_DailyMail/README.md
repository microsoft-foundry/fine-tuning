# Supervised Fine-Tuning with CNN/DailyMail News Summarization Dataset

This cookbook demonstrates how to fine-tune language models using **Supervised Fine-Tuning (SFT)** with the CNN/DailyMail News Text Summarization dataset on Microsoft Foundry.

## Overview

Supervised Fine-Tuning (SFT) is a technique for training language models to perform specific tasks by learning from labeled examples. This cookbook uses the CNN/DailyMail dataset, which contains over 300,000 English news articles with corresponding summaries written by professional journalists at CNN and Daily Mail.

This dataset is ideal for training models to:

- Generate concise, accurate summaries of news articles
- Extract key information from long-form text
- Understand journalistic writing styles
- Produce professional-quality content summaries

## Dataset Information

**Source**: [CNN/DailyMail on Kaggle](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

**Size**: 2,221 article-summary pairs (curated subset for demonstration)
- Training set: 1,992 examples (90%)
- Validation set: 229 examples (10%)

> **Note**: This is a carefully curated subset of the full CNN/DailyMail dataset, optimized for learning and demonstration purposes. The full dataset contains over 300,000 examples.
>
> **RAI Compliance**: The datasets have been pre-filtered to pass Azure AI content safety checks. News articles containing content flagged for violence, hate speech, or other sensitive content have been removed to ensure smooth fine-tuning job execution.

**What the Data Contains**:
The CNN/DailyMail dataset consists of news articles paired with professionally-written highlights/summaries. Each example includes:
- **Article**: The full text of the news article (average ~781 tokens)
- **Summary**: A concise highlight written by the article's journalist (average ~56 tokens)

**Task**: Abstractive Summarization
The model learns to generate concise summaries that:
- **Capture key information**: Main events, people, and facts from the article
- **Maintain factual accuracy**: Stay true to the source material
- **Use concise language**: Distill content into brief, readable summaries
- **Follow journalistic style**: Match professional news summary conventions

## What You'll Learn

This cookbook teaches you how to:

1. Set up your Microsoft Foundry environment for supervised fine-tuning
2. Prepare and format news summarization data in JSONL format
3. Upload datasets to Microsoft Foundry
4. Create and configure a supervised fine-tuning job
5. Monitor training progress
6. Download service metrics and assess training convergence

## Prerequisites

- Azure subscription with Microsoft Foundry project, you must have **Azure AI User** role
- An Entra ID identity available to `DefaultAzureCredential`; for local development, run `az login`
- Python 3.9 or higher
- Familiarity with Jupyter notebooks
- CNN/DailyMail dataset CSV files (download from Kaggle)

## Supported Models

Find the supported fine-tuning models in Microsoft Foundry [here](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/fine-tuning-overview?view=foundry-classic). Model availability may vary by region. Check the [Azure OpenAI model availability](https://learn.microsoft.com/azure/ai-services/openai/concepts/models) page for the most current regional support.

## Files in This Cookbook

- **README.md**: This file - comprehensive documentation
- **requirements.txt**: Python dependencies required for the cookbook
- **training.jsonl**: Training dataset (1,992 article-summary pairs, RAI-filtered)
- **validation.jsonl**: Validation dataset (229 article-summary pairs, RAI-filtered)
- **sft_cnn_dailymail.ipynb**: Step-by-step notebook implementation

## Quick Start

### 1. Prepare Your Dataset

The training and validation JSONL files are already provided in this directory. If you want to create your own or use a different subset:

1. Download the CNN/DailyMail dataset from Kaggle:
   https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

2. Convert the CSV files to JSONL format following the structure shown in the "Dataset Format" section below

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Authenticate

Run `az login` before executing the notebook. Authentication uses
`DefaultAzureCredential` with `AIProjectClient`; API keys are not used.

Configure the Microsoft Foundry project before running:

```properties
AZURE_AI_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
AZURE_AI_REGION=<region>
```

### 4. Run the Notebook

Open sft_cnn_dailymail.ipynb and follow the step-by-step instructions.

## Dataset Format

The supervised fine-tuning format follows the Microsoft Foundry chat completion structure:

```json
{
  \"messages\": [
    {\"role\": \"system\", \"content\": \"You are a news summarization assistant. Create concise summaries of news articles.\"},
    {\"role\": \"user\", \"content\": \"Summarize this article:\\n\\n[Full article text...]\"},
    {\"role\": \"assistant\", \"content\": \"[Professional summary...]\"}
  ]
}
```

Each training example contains:
- **system message**: Instructions for the model's behavior
- **user message**: The news article to summarize
- **assistant message**: The professional summary (ground truth)

## Training Configuration

The cookbook uses the following hyperparameters:

- **Model**: Muse-Glimmer-30B, version 1
- **Training type**: GlobalStandard
- **Epochs**: 1
- **Batch Size**: 1
- **Learning Rate Multiplier**: 1.0

These can be adjusted based on your specific requirements.

## Expected Outcomes

This run is intentionally training-only. It preserves all 1,992 training rows
and all 229 validation rows, then bases its conclusion on service-reported
validation loss and validation token accuracy. It does not deploy either the
base model or fine-tuned model and does not run inference. If the service omits
token accuracy, the notebook reports the loss trend and marks the two-metric
conclusion as qualified rather than fabricating the missing metric.

## Monitoring

The notebook includes sections for:

- Real-time training progress monitoring
- Validation loss tracking
- Validation token-accuracy tracking
- Terminal evidence capture under `outputs/loom-model-runs/`

## Next Steps

After completing this cookbook, you can:

1. Review the service validation metrics and terminal job status
2. Experiment with different hyperparameters in a separate controlled run
3. Add deployment and held-out inference evaluation only when explicitly needed

## Troubleshooting

### Common Issues

**File Not Ready Error**
- Ensure uploaded files are fully processed before starting the job
- Wait a few minutes after upload for processing to complete
- Check file status in AI Foundry portal

**Training Job Fails Immediately**
- Verify data format matches the JSONL schema shown above
- Ensure all required fields (messages array with system/user/assistant) are present
- Check that no examples exceed token limits (typically 4096 tokens)

**RAI Content Safety Check Failed**
- Azure AI performs content safety checks on all training data
- News datasets often contain content flagged for violence, hate, or sensitive topics
- The error message will list specific line numbers that failed
- Remove flagged lines from your dataset and re-upload
- The provided datasets are pre-filtered to pass RAI checks

**Authentication Error**
- Run `az login` to refresh your Azure credentials
- Verify your subscription has access to Azure AI Foundry
- Check that you have the **Azure AI User** role on the project

**Quota Exceeded**
- Request additional quota in Azure Portal → Azure OpenAI → Quotas
- Try a different Azure region with available capacity
- Use a smaller model variant if available

**Model Deployment Fails**
- Ensure you have **Cognitive Services OpenAI User** role
- Check that the fine-tuned model completed training successfully
- Verify deployment name doesn't conflict with existing deployments

## References

- [CNN/DailyMail Dataset on Kaggle](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
- [Azure OpenAI Fine-tuning Documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning)
- [Azure AI Foundry Documentation](https://learn.microsoft.com/azure/ai-studio/)

---

**Ready to get started?** Download the dataset, generate the JSONL files, and open the notebook!
