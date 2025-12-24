# DPO Fine-Tuning with Intel Orca DPO Pairs Dataset

This cookbook demonstrates how to fine-tune language models using Direct Preference Optimization (DPO) with the Intel Orca DPO Pairs dataset on Azure AI.

## Overview

Direct Preference Optimization (DPO) is a technique for training language models to prefer high-quality responses over lower-quality alternatives. This cookbook uses the Intel Orca DPO Pairs dataset, which contains approximately 12,900 preference pairs covering diverse tasks including:

- Mathematical reasoning
- Reading comprehension
- Logical reasoning
- Text summarization
- General Q&A

## Dataset Information

**Source**: [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs)

**Size**: ~3,000 preference pairs (curated subset for demonstration)
- Training set: ~2,750 examples (90%)
- Validation set: ~284 examples (10%)

> **Note**: This is a carefully curated subset of the full Intel Orca DPO Pairs dataset, optimized for learning and demonstration purposes. The full dataset contains ~12,900 examples and can be downloaded from [Hugging Face](https://huggingface.co/datasets/Intel/orca_dpo_pairs).

**What the Data Contains**:
The Intel Orca DPO Pairs dataset consists of instruction-response pairs across multiple domains including mathematics, reasoning, comprehension, and general knowledge. Each example includes the same input prompt with two different responses - one preferred (high-quality) and one non-preferred (lower-quality).

**Preference Optimization**:
DPO optimizes the model to favor responses that are:
- **More accurate**: Factually correct and precise answers
- **Better reasoned**: Clear logical progression and explanation
- **More helpful**: Directly addresses the user's question
- **Properly formatted**: Well-structured and complete responses

The model learns to increase the probability of generating preferred responses while decreasing the probability of non-preferred ones, without requiring explicit reward modeling or reinforcement learning.

**Format**: Each example contains:
- input: System and user messages (the prompt)
- preferred_output: The high-quality chosen response
- non_preferred_output: The lower-quality rejected response

## What You'll Learn

This cookbook teaches you how to:

1. Set up your Azure AI environment for DPO fine-tuning
2. Prepare and format DPO training data in JSONL format
3. Upload datasets to Azure AI
4. Create and configure a DPO fine-tuning job
5. Monitor training progress
6. Deploy and inference your fine-tuned model

## Prerequisites

- Azure subscription with Microsoft foundry project, you must have **Azure AI User** role
- Python 3.9 or higher
- Familiarity with Jupyter notebooks

## Supported Models

DPO fine-tuning in Azure AI Foundry supports the following GPT models:

- **gpt-4o**: Latest GPT-4 Optimized model
- **gpt-4o-mini**: Cost-efficient variant of GPT-4o
- **gpt-4.1**: GPT-4.1 base model
- **gpt-4.1-mini**: Compact version of GPT-4.1

> **Note**: Model availability may vary by region. Check the [Azure OpenAI model availability](https://learn.microsoft.com/azure/ai-services/openai/concepts/models) page for the most current regional support.

## Files in This Cookbook

- README.md: This file - comprehensive documentation
- requirements.txt: Python dependencies required for the cookbook
- training.jsonl: Training dataset
- validation.jsonl: Validation dataset
- dpo_finetuning_intel_orca.ipynb: Step-by-step notebook implementation

## Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a .env file in this directory with your Azure credentials:

```
AZURE_AI_PROJECT_ENDPOINT=<your-endpoint> 
MODEL_NAME=<your-gpt-model-name>
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_AOAI_ACCOUNT=<your-foundry-account-name>
```

### 3. Run the Notebook

Open dpo_finetuning_intel_orca.ipynb and follow the step-by-step instructions.

## Dataset Format

The DPO format follows the Azure AI Projects SDK structure:

```json
{
  \"input\": {
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"What is the capital of France?\"}
    ]
  },
  \"preferred_output\": [
    {\"role\": \"assistant\", \"content\": \"The capital of France is Paris.\"}
  ],
  \"non_preferred_output\": [
    {\"role\": \"assistant\", \"content\": \"I think it's London.\"}
  ]
}
```

## Training Configuration

The cookbook uses the following hyperparameters:

- **Model**: gpt-4o-mini (or gpt-4o)
- **Training Type**: Standard
- **Epochs**: 3
- **Batch Size**: 1
- **Learning Rate Multiplier**: 1.0

These can be adjusted based on your specific requirements.

## Expected Outcomes

After fine-tuning with DPO, your model should:

- Generate more accurate and helpful responses
- Better align with human preferences
- Improve performance on reasoning tasks
- Provide more reliable answers across diverse topics

## Monitoring and Evaluation

The notebook includes sections for:

- Real-time training progress monitoring
- Validation loss tracking
- Model performance evaluation
- Inference examples with the fine-tuned model

## Troubleshooting

### Common Issues

**File Not Ready Error**
- Ensure uploaded files are fully processed before starting the job
- Use wait_for_processing() method

**Training Failure**
- Check data format matches expected schema
- Verify all required fields are present
- Ensure no examples exceed token limits

**Authentication Error**
- Verify your Azure credentials are correct
- Check that your subscription has access to Azure AI Studio
