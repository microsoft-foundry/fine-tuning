# DPO Fine-Tuning with Intel Orca DPO Pairs Dataset

This cookbook demonstrates how to fine-tune language models using Direct Preference Optimization (DPO) with the Intel Orca DPO Pairs dataset on Microsoft foundry.

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

1. Set up your Microsoft foundry environment for DPO fine-tuning
2. Prepare and format DPO training data in JSONL format
3. Upload datasets to Microsoft foundry
4. Evaluate base model performance before fine-tuning
5. Create and configure a DPO fine-tuning job
6. Monitor training progress
7. Deploy and inference your fine-tuned model
8. Evaluate fine-tuned model and compare improvements using Azure AI Evaluation SDK

## Prerequisites

- Azure subscription with Microsoft foundry project, you must have **Azure AI User** role
- Python 3.9 or higher
- Familiarity with Jupyter notebooks

## Supported Models

Find the supported DPO fine-tuning models in Microsoft foundry [here](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/fine-tuning-overview?view=foundry-classic). Model availability may vary by region. Check the [Azure OpenAI model availability](https://learn.microsoft.com/azure/ai-services/openai/concepts/models) page for the most current regional support.

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

Copy the file `.env.template` (located in this folder), and save it as file named `.env`. Enter appropriate values for the environment variables used for the job you want to run.

```
# Required for DPO Fine-Tuning
MICROSOFT_FOUNDRY_PROJECT_ENDPOINT=<your-endpoint> 
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_AOAI_ACCOUNT=<your-foundry-account-name>
MODEL_NAME=<your-base-model-name>

# Required for Model Evaluation
AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
AZURE_OPENAI_KEY=<your-azure-openai-api-key>
DEPLOYMENT_NAME=<your-deployment-name>
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

- **Model**: gpt-4.1-mini
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
- Model performance evaluation with Azure AI Evaluation SDK
- Inference examples with the fine-tuned model

### Evaluation Metrics for DPO Fine-Tuning

This cookbook uses the **Azure AI Evaluation SDK** to comprehensively assess model quality before and after DPO fine-tuning. The Azure AI Evaluation SDK provides production-ready evaluators that use GPT models as judges to score your model outputs across multiple dimensions.

**3 Key Metrics for DPO Evaluation**:

1. **Coherence (1-5)**: Measures logical flow and structure - ensures responses are well-organized and make sense
2. **Fluency (1-5)**: Assesses grammatical correctness and natural language quality - validates human-like writing
3. **Groundedness (1-5)**: Checks factual accuracy against provided context - prevents hallucinations and unsupported claims

**Why These Metrics Matter for DPO**:
- DPO trains models to prefer high-quality responses over low-quality alternatives
- These metrics validate that the fine-tuned model generates responses that are both accurate and aligned with human preferences
- Comparing pre- and post-fine-tuning scores demonstrates the effectiveness of DPO training

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
- Check that your subscription has access to Microsoft Foundry
