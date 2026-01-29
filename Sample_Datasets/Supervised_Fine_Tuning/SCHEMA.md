# Supervised Fine-Tuning (SFT) Data Schema

This document describes the JSONL format required for Supervised Fine-Tuning jobs in Azure AI Foundry.

## Format Overview

SFT uses a **chat completion format** where each training example is a conversation with messages from different roles.

## Schema Definition

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<system prompt - optional>"
    },
    {
      "role": "user", 
      "content": "<user input/question>"
    },
    {
      "role": "assistant",
      "content": "<desired model response>"
    }
  ]
}
```

## Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array | ✅ Yes | Array of message objects representing the conversation |
| `messages[].role` | string | ✅ Yes | One of: `"system"`, `"user"`, `"assistant"` |
| `messages[].content` | string | ✅ Yes | The text content for this message |

## Role Descriptions

- **system**: Sets the behavior, tone, or context for the assistant (optional but recommended)
- **user**: The input or question from the end user
- **assistant**: The desired response the model should learn to produce

## Example Records

### Text Example (GSM8K - Math)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Reply user's question as accurately as possible."
    },
    {
      "role": "user",
      "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    },
    {
      "role": "assistant",
      "content": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
    }
  ]
}
```

### Tool Calling Example
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to stock market tools."
    },
    {
      "role": "user",
      "content": "What is the current price of Microsoft stock?"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_current_stock_price",
            "arguments": "{\"symbol\": \"MSFT\"}"
          }
        }
      ]
    }
  ]
}
```

## Validation Checklist

- [ ] File is valid JSONL (one JSON object per line)
- [ ] Each record has a `messages` array
- [ ] Each message has `role` and `content` fields
- [ ] At least one `user` and one `assistant` message per record
- [ ] No trailing commas in JSON objects

## Datasets in this Folder

| Dataset | Description | Records |
|---------|-------------|---------|
| [Text-GSM8K](./Text-GSM8K/) | Grade school math problems | ~7,500 |
| [Multimodal-chartqa](./Multimodal-chartqa/) | Chart interpretation with images | ~18,000 |
| [Tool-Calling](./Tool-Calling/) | Stock market function calling | ~500 |
