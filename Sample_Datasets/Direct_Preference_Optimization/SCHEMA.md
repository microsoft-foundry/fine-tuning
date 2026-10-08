# Direct Preference Optimization (DPO) Data Schema

This document describes the JSONL format required for Direct Preference Optimization jobs in Azure AI Foundry.

## Format Overview

DPO uses a **preference pair format** where each training example contains an input prompt along with a preferred response and a non-preferred response. The model learns to favor the preferred output.

## Schema Definition

```json
{
  "input": {
    "messages": [
      {
        "role": "system",
        "content": "<system prompt - optional>"
      },
      {
        "role": "user",
        "content": "<user input/question>"
      }
    ]
  },
  "preferred_output": [
    {
      "role": "assistant",
      "content": "<higher quality response>"
    }
  ],
  "non_preferred_output": [
    {
      "role": "assistant",
      "content": "<lower quality response>"
    }
  ]
}
```

## Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | object | ✅ Yes | Contains the conversation context/prompt |
| `input.messages` | array | ✅ Yes | Array of system and user messages |
| `preferred_output` | array | ✅ Yes | The preferred assistant response(s) |
| `non_preferred_output` | array | ✅ Yes | The non-preferred assistant response(s) |

## What Makes a Response "Preferred"?

The preferred output should be better than the non-preferred output in one or more of:
- **Accuracy**: More factually correct
- **Helpfulness**: More directly addresses the user's need
- **Safety**: More appropriate and harmless
- **Style**: Better matches desired tone, formality, or format
- **Reasoning**: Shows clearer logic or better explanation

## Example Record (orca_dpo_pairs)

```json
{
  "input": {
    "messages": [
      {
        "role": "system",
        "content": "You are an AI assistant that follows instructions extremely well."
      },
      {
        "role": "user",
        "content": "Multi-choice question: What is the sentiment of the following review?\nThe food was bland and the service was slow.\nOptions:\n- negative\n- positive"
      }
    ]
  },
  "preferred_output": [
    {
      "role": "assistant",
      "content": "To determine the sentiment of this review, let's analyze the key descriptors:\n\n1. \"bland\" - This is a negative word describing food that lacks flavor\n2. \"slow\" - This is a negative word describing poor service speed\n\nBoth descriptors are negative, and there are no positive elements mentioned.\n\nThe answer is: **negative**"
    }
  ],
  "non_preferred_output": [
    {
      "role": "assistant",
      "content": "The answer is negative."
    }
  ]
}
```

**Why preferred is better**: The preferred response explains the reasoning, while the non-preferred just states the answer without justification.

## Validation Checklist

- [ ] File is valid JSONL (one JSON object per line)
- [ ] Each record has `input`, `preferred_output`, and `non_preferred_output`
- [ ] `input.messages` contains at least one `user` message
- [ ] Both outputs contain at least one `assistant` message
- [ ] Preferred output is meaningfully different from non-preferred

## Datasets in this Folder

| Dataset | Description | Records |
|---------|-------------|---------|
| [orca_dpo_pairs](./orca_dpo_pairs/) | OpenOrca preference pairs for alignment | ~3,000 |
