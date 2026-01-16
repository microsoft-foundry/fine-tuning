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

**Size**: 2,504 article-summary pairs (curated subset for demonstration)
- Training set: 2,255 examples (90%)
- Validation set: 249 examples (10%)

> **Note**: This is a carefully curated subset of the full CNN/DailyMail dataset, optimized for learning and demonstration purposes. The full dataset contains over 300,000 examples.

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
6. Deploy and test your fine-tuned model

## Prerequisites

- Azure subscription with Microsoft Foundry project, you must have **Azure AI User** role
- Python 3.9 or higher
- Familiarity with Jupyter notebooks
- CNN/DailyMail dataset CSV files (download from Kaggle)

## Supported Models

Find the supported DPO fine-tuning models in Microsoft foundry [here](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/fine-tuning-overview?view=foundry-classic). Model availability may vary by region. Check the [Azure OpenAI model availability](https://learn.microsoft.com/azure/ai-services/openai/concepts/models) page for the most current regional support.

## Files in This Cookbook

- **README.md**: This file - comprehensive documentation
- **requirements.txt**: Python dependencies required for the cookbook
- **training.jsonl**: Training dataset (2,255 article-summary pairs)
- **validation.jsonl**: Validation dataset (249 article-summary pairs)
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

### 3. Set Up Environment Variables

Copy the file `.env.template` (located in this folder), and save it as file named `.env`. Enter appropriate values for the environment variables used for the job you want to run.

```
MICROSOFT_FOUNDRY_PROJECT_ENDPOINT=<your-endpoint> 
MODEL_NAME=<your-gpt-model-name>
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_AOAI_ACCOUNT=<your-foundry-account-name>
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

- **Model**: gpt-4.1
- **Epochs**: 3
- **Batch Size**: 1
- **Learning Rate Multiplier**: 1.0

These can be adjusted based on your specific requirements.

## Expected Outcomes

After fine-tuning with the CNN/DailyMail dataset, your model should:

- Generate concise, accurate news summaries
- Better understand journalistic writing styles
- Improve factual accuracy in summarizations
- Produce summaries similar to professional journalist output
- Handle various news topics and article lengths

## Monitoring

The notebook includes sections for:

- Real-time training progress monitoring
- Validation loss tracking
- Model deployment and testing

## Next Steps

After completing this cookbook, you can:

1. Fine-tune on your own news or document summarization data
2. Experiment with different hyperparameters
3. Combine with retrieval-augmented generation (RAG)
4. Deploy to production applications
5. Integrate into content management systems

## References

- [CNN/DailyMail Dataset on Kaggle](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
- [Azure OpenAI Fine-tuning Documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning)

---

**Ready to get started?** Download the dataset, generate the JSONL files, and open the notebook!
