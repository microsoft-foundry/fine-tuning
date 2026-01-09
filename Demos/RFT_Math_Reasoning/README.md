# Reinforcement Fine-Tuning with OpenR1-Math-220k Dataset

This cookbook demonstrates how to fine-tune language models using **Reinforcement Fine-Tuning (RFT)** with the OpenR1-Math-220k dataset on Azure AI. This dataset contains 220,000 advanced mathematical reasoning problems with verified step-by-step solutions, making it ideal for teaching models complex mathematical problem-solving.

## Overview

Reinforcement Fine-Tuning (RFT) is a powerful technique for training language models on tasks where:
- Multiple valid solution paths exist
- Correctness can be verified automatically
- Step-by-step reasoning is crucial
- The task requires structured, multi-hop thinking

This cookbook uses the **OpenR1-Math-220k dataset**, which contains advanced mathematics problems from college-level and competition mathematics. For RFT, we use only the problem statements and final answers extracted from this dataset. The problems cover diverse mathematical domains including algebra, calculus, geometry, and number theory.

## Dataset Information

**Source**: [OpenR1-Math-220k on Kaggle](https://www.kaggle.com/datasets/alejopaullier/openr1-math-220k) | [Hugging Face](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)

**Dataset Statistics**:
- **Full dataset**: 220,000 mathematical reasoning problems (7 parquet files, 1.45 GB total)
- **Source parquet**: train-00000-of-00007.parquet (13,391 problems with 27,614 verified reasoning traces)
- **Training set**: 2,661 examples (one entry per problem)
- **Validation set**: 565 examples (one entry per problem)

> **Important**: 
> - The RFT dataset contains processed data extracted from verified parquet files
> - Each training example contains **prompts only** (system + user) and a ground truth answer


**What the Data Contains**:
The RFT dataset consists of advanced mathematical problems with ground truth answers. Each training example includes:
- **System prompt**: Instructions for mathematical problem-solving behavior
- **Problem statement**: Complex mathematics question requiring multi-step reasoning
- **Ground truth answer**: Final answer extracted from verified solutions, used by the grader for scoring

**Note**: The RFT format contains only prompts (system + user) and answers. The model learns to generate step-by-step reasoning through reinforcement learning, not by mimicking provided reasoning traces.

**Problem Domains Covered**:
- Algebra and polynomial factorization
- Calculus and optimization
- Probability theory and expected values
- Geometry and trigonometry
- Number theory and combinatorics
- Linear algebra and matrix operations
- Complex numbers and abstract mathematics

**Task Complexity**: 
- Advanced multi-step mathematical reasoning
- Problems require abstract algebraic manipulation
- Multi-hop logical reasoning (typically 5-15 reasoning steps)
- Advanced mathematical concepts beyond basic arithmetic
- The model must generate detailed reasoning chains (averaging 8k tokens, complex problems up to 16k tokens)

**Why RFT is Perfect for This Task**:
- **Multiple valid solution paths**: Many ways to solve each problem
- **Verifiable correctness**: Mathematical answers can be automatically checked
- **Exploration-based learning**: Model discovers effective reasoning strategies through reinforcement
- **Quality over memorization**: Model learns reasoning patterns, not just answers

## What You'll Learn

This cookbook teaches you how to:

1. Understand the RFT dataset format (prompts + ground truth answers)
2. Set up your Azure AI environment for RFT
3. Create a grading function to evaluate mathematical reasoning quality
4. Configure and launch an RFT fine-tuning job
5. Monitor training progress and model performance
6. Deploy and test your fine-tuned mathematical reasoning model

**Note**: The RFT-formatted dataset files (training_rft.jsonl and validation_rft.jsonl) are already provided. If you want to prepare your own dataset from the original Kaggle files, you'll need to extract problems and answers into the RFT format shown in the Dataset Format section.

## Prerequisites

- Azure subscription with Microsoft Foundry project (requires **Azure AI User** role)
- Python 3.9 or higher
- Familiarity with Jupyter notebooks
- Kaggle account to download the OpenR1-Math-220k dataset
- Understanding of basic mathematical concepts

## Supported Models

RFT in Azure AI Foundry supports the following models:

- **o4-mini**
- **gpt-5 (PrPr)**


> **Note**: Model availability may vary by region. Check the [Azure OpenAI model availability](https://learn.microsoft.com/azure/ai-services/openai/concepts/models) page for current regional support.

## Files in This Cookbook

- **README.md**: This file - comprehensive documentation
- **requirements.txt**: Python dependencies required for the cookbook
- **rft_math_reasoning.ipynb**: Step-by-step notebook implementation
- **training_rft.jsonl**: Training dataset (2,661 examples in RFT format)
- **validation_rft.jsonl**: Validation dataset (565 examples in RFT format)

## Quick Start


### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Prepare the Dataset

You can use the training and validation dataset as is from this directory or you can prepare your own dataset.

### 3. Set Up Environment Variables

Create a `.env` file in the root of this directory with your Azure credentials:

```env
# Required for RFT Fine-Tuning
AZURE_AI_PROJECT_ENDPOINT=<your-endpoint>
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_AOAI_ACCOUNT=<your-foundry-account-name>
MODEL_NAME=<your-base-model-name>
```

### 4. Run the Notebook

Open `rft_math_reasoning.ipynb` and follow the step-by-step instructions.

## Dataset Format

The RFT format for mathematical reasoning follows this structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are a mathematical reasoning expert. Solve problems with detailed step-by-step thinking and provide final answers in \\boxed{} format. Show all intermediate calculations and explain your reasoning clearly.\n\nCall a scalene triangle K disguisable if there exists a triangle K′ similar to K with two shorter sides precisely as long as the two longer sides of K, respectively. Call a disguisable triangle integral if the lengths of all its sides are integers. (a) Find the side lengths of the integral disguisable triangle with the smallest possible perimeter. (b) Let K be an arbitrary integral disguisable triangle for which no smaller integral disguisable triangle similar to it exists. Prove that at least two side lengths of K are perfect squares."
    }
  ],
  "answer": "9"
}
```

Each training example contains:
- **messages**: Array with exactly 1 message:
  - **user message**: Combined system prompt and mathematical problem statement. The system instructions are included at the beginning of the user content, followed by the actual problem.
- **answer**: Ground truth final answer (used by the grader for verification)

**Important**: Unlike Supervised Fine-Tuning (SFT), RFT format does NOT include an assistant message. The model learns to generate solutions through reinforcement signals from the grader, which compares the model's output against the ground truth answer field.

## Training Configuration

Recommended hyperparameters for RFT with mathematical reasoning:

- **Model**: o4-mini
- **Epochs**: 2 (prevents overfitting while allowing reinforcement learning)
- **Batch Size**: 1 (limited by memory constraints with long sequence lengths)
- **Learning Rate Multiplier**: 1.0 (standard learning rate for RFT)

These can be adjusted based on your computational resources and specific requirements.

## Grading Mathematical Solutions

The grading function for RFT is a deterministic Python function that evaluates:

1. **Answer Extraction**: Extracts the final answer from the model's `\boxed{}` notation
2. **Answer Normalization**: Handles different formats (comma-separated, space-separated, LaTeX expressions)
3. **Answer Correctness**: Compares normalized model answer against ground truth

The grader returns:
- **1.0**: Model answer matches ground truth (after normalization)
- **0.0**: Model answer does not match or is missing/malformed

This binary reward signal guides the model to learn effective reasoning strategies that produce correct final answers.

## Expected Outcomes

After fine-tuning with RFT on OpenR1-Math-220k, your model should:

- Generate detailed step-by-step mathematical reasoning chains with clear intermediate steps
- Solve college-level and competition mathematics problems across diverse domains
- Produce properly formatted answers with `\boxed{}` notation for grader verification
- Demonstrate significantly improved performance on multi-hop mathematical reasoning compared to the base model


## Additional Resources

- [Azure OpenAI Fine-Tuning Documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning)
- [OpenR1-Math-220k dataset](https://www.kaggle.com/datasets/alejopaullier/openr1-math-220k)

---
