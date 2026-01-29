# Reinforcement Fine-Tuning (RFT) Data Schema

This document describes the JSONL format required for Reinforcement Fine-Tuning jobs in Azure AI Foundry.

## Format Overview

RFT uses a **question + reference answer format** where each training example contains a prompt and a verifiable correct answer. A grader evaluates the model's responses during training.

## Schema Definition

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<question or problem>"
    }
  ],
  "reference_answer": "<correct answer for grading>"
}
```

## Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array | ✅ Yes | Array containing the user prompt |
| `messages[].role` | string | ✅ Yes | Should be `"user"` for the question |
| `messages[].content` | string | ✅ Yes | The question or problem to solve |
| `reference_answer` | string | ✅ Yes | The correct answer used by the grader |

## Graders

RFT requires a **grader** to evaluate model responses. Common grader types:

| Grader Type | Use Case | Example |
|-------------|----------|---------|
| `string_check` | Exact or partial string matching | Multiple choice answers (A, B, C, D) |
| `model_grader` | LLM-based evaluation | Open-ended responses |
| `python_grader` | Custom Python logic | Math problems, code execution |
| `multi_grader` | Combine multiple graders | String check + similarity score |

## Example Records

### Medical Multiple Choice (MedMCQ)
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma:\n- A. Hyperplasia\n- B. Hypertrophy\n- C. Atrophy\n- D. Dysplasia"
    }
  ],
  "reference_answer": "C"
}
```

**Grader**: `string_check` - Verifies the model outputs "C"

### Legal Clause Matching (ClauseMatching)
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Identify the clause type in this contract excerpt: 'The Licensee shall not assign, sublicense, or transfer any rights under this Agreement without prior written consent.'"
    }
  ],
  "reference_answer": "Assignment/Sublicense Restriction"
}
```

**Grader**: `multi_grader` - Combines string check with semantic similarity

## Grader Configuration Examples

### String Check Grader (`stringcheck_grader.json`)
```json
{
  "type": "string_check",
  "name": "answer_checker",
  "input": "{{item.reference_answer}}",
  "reference": "{{sample.output_text}}",
  "operation": "contains"
}
```

### Model Grader (`model_grader.json`)
```json
{
  "type": "score_model",
  "name": "quality_grader",
  "model": "gpt-4o",
  "input": [
    {"role": "system", "content": "Grade the response..."},
    {"role": "user", "content": "Question: {{item.messages[0].content}}\nResponse: {{sample.output_text}}"}
  ]
}
```

## When to Use RFT

RFT works best when:
- ✅ There are **clear right/wrong answers** (objective domains)
- ✅ **Expert evaluators would agree** on correctness
- ✅ The model already shows **some competency** on the task
- ✅ **Lucky guessing is difficult** (complex reasoning required)

**Good domains**: Mathematics, chemistry, physics, law, medicine

## Validation Checklist

- [ ] File is valid JSONL (one JSON object per line)
- [ ] Each record has `messages` and `reference_answer`
- [ ] Reference answers are consistent in format
- [ ] A compatible grader configuration is provided
- [ ] Grader can correctly evaluate the reference answers

## Datasets in this Folder

| Dataset | Description | Grader Type | Records |
|---------|-------------|-------------|---------|
| [ClauseMatching](./ClauseMatching/) | Legal contract clause identification | multi_grader | ~500 |
| [MedMCQ](./MedMCQ/) | Medical entrance exam questions | string_check | ~1,000 |
