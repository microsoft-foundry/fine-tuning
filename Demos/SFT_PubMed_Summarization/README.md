# Supervised Fine-Tuning with PubMed Medical Research Summarization Dataset

This cookbook demonstrates how to fine-tune language models using **Supervised Fine-Tuning (SFT)** with the PubMed Medical Research Summarization dataset on Microsoft Foundry.

## Overview

Supervised Fine-Tuning (SFT) is a technique for training language models to perform specific tasks by learning from labeled examples. This cookbook uses the PubMed dataset, which contains scientific research articles from the PubMed database paired with their corresponding abstracts. The dataset covers a wide range of topics across various scientific and medical disciplines.

Medical research articles often contain complex scientific information, detailed methodologies, and specialized terminology. This dataset is ideal for training models to:

- Generate concise, accurate abstracts from full medical research articles
- Extract key findings and methodologies from scientific papers
- Understand medical and scientific terminology
- Produce professional-quality research summaries
- Maintain scientific accuracy in summarization

## Dataset Information

**Source**: [PubMed Article Summarization Dataset on Kaggle](https://www.kaggle.com/datasets/thedevastator/pubmed-article-summarization-dataset)

**Size**: 6,655 article-abstract pairs (curated subset for demonstration)
- Training set: 4,999 examples
- Validation set: 1,656 examples

> **Note**: This is a carefully curated subset of the full PubMed dataset (~4% of the original 119,924 training examples), optimized for efficient training and demonstration purposes while maintaining representative coverage of medical research topics and staying within GitHub's file size limits.

**What the Data Contains**:
The PubMed dataset consists of medical and scientific research articles paired with professionally-written abstracts. Each example includes:
- **Article**: The full text of a scientific article from PubMed
- **Abstract**: A concise summary of the main findings, methodology, and conclusions

**Format Example**

Each example in the JSONL files follows the Azure OpenAI chat completion format:

```json
{
  \"messages\": [
    {
      \"role\": \"system\",
      \"content\": \"You are a medical research summarization assistant. Create concise, accurate abstracts of medical research articles that capture the key findings and methodology.\"
    },
    {
      \"role\": \"user\",
      \"content\": \"Summarize this medical research article:\n\n[full article text]\"
    },
    {
      \"role\": \"assistant\",
      \"content\": \"[generated abstract]\"
    }
  ]
}
```

Each training example contains:
- **system message**: Instructions for the model's behavior and domain expertise
- **user message**: The medical research article to summarize
- **assistant message**: The professionally-written abstract (ground truth)

**Task**: Scientific Abstract Generation
The model learns to generate concise abstracts that:
- **Capture key findings**: Main results, conclusions, and significance
- **Describe methodology**: Research methods and study design
- **Maintain scientific accuracy**: Use precise medical/scientific terminology
- **Follow academic style**: Match professional research abstract conventions

## What You'll Learn

This cookbook teaches you how to:

1. Set up your Microsoft Foundry environment for supervised fine-tuning
2. Prepare and format medical research data in JSONL format
3. Upload datasets to Microsoft Foundry
4. Create and configure a supervised fine-tuning job
5. Monitor training progress and review metrics
6. Deploy and test your fine-tuned model

## Prerequisites

- Azure subscription with Microsoft Foundry project; you must have **Azure AI User** role
- Python 3.9 or higher
- Familiarity with Jupyter notebooks
- Basic understanding of medical/scientific literature (helpful but not required)

## Supported Models

Find the supported DPO fine-tuning models in Microsoft foundry [here](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/fine-tuning-overview?view=foundry-classic). Model availability may vary by region. Check the [Azure OpenAI model availability](https://learn.microsoft.com/azure/ai-services/openai/concepts/models) page for the most current regional support.

## Files in This Cookbook

- **README.md**: This file - comprehensive documentation
- **requirements.txt**: Python dependencies required for the cookbook
- **training.jsonl**: Training dataset (4,999 article-abstract pairs)
- **validation.jsonl**: Validation dataset (1,656 article-abstract pairs)
- **sft_pubmed_summarization.ipynb**: Step-by-step notebook implementation

## Quick Start

### 1. Prepare Your Dataset

The training and validation JSONL files are already provided in this directory. If you want to create your own or use a different subset:

1. Download the PubMed dataset from Kaggle:
   https://www.kaggle.com/datasets/thedevastator/pubmed-article-summarization-dataset

2. Convert the CSV files to JSONL format following the structure shown above

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the file `.env.template` (located in this folder), and save it as file named `.env`. Enter appropriate values for the environment variables used for the job you want to run.

```
MICROSOFT_FOUNDRY_PROJECT_ENDPOINT=<your-endpoint> 
MODEL_NAME=gpt-4.1
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_AOAI_ACCOUNT=<your-foundry-account-name>
```

### 4. Run the Notebook

Open sft_pubmed_summarization.ipynb and follow the step-by-step instructions.

## Training Configuration

The cookbook uses the following hyperparameters:

- **Model**: gpt-4.1
- **Epochs**: 3
- **Batch Size**: 1
- **Learning Rate Multiplier**: 1.0
- **Suffix**: pubmed-summarization

These can be adjusted based on your specific requirements and dataset characteristics.

## Next Steps

After completing this cookbook, you can:

1. Fine-tune on your own medical or scientific literature
2. Experiment with different hyperparameters for specialized medical domains
3. Deploy to production medical research applications


## References

- [Fine-Tuning Guide](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning)
- [PubMed Dataset on Kaggle](https://www.kaggle.com/datasets/thedevastator/pubmed-article-summarization-dataset)
