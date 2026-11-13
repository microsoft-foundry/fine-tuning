# Supervised Fine-Tuning (SFT) Data Schema

This document describes the union of chat formats used by these sample datasets.
It is not a claim that every fine-tuning model accepts every modality. Check the
selected model's [text SFT](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning),
[vision SFT](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning-vision)
and [tool-calling SFT](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning-functions)
requirements before upload. These sample formats do not add audio/video support
or establish support in a programmed-training renderer.

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
| `messages[].role` | string | Yes | `"system"`, `"user"`, `"assistant"`, or `"tool"` |
| `messages[].content` | string, content-part array, or conditional null | Conditional | See the role-specific union below; omitted/null only for an assistant with nonempty `tool_calls` |
| `messages[].tool_calls` | array | On assistant tool-call turns | Nonempty function calls: `id`, `type: "function"`, `function.name`, JSON-encoded string `function.arguments` |
| `messages[].tool_call_id` | string | On tool replies | References the preceding assistant call's `id` |
| `tools` | array | For examples with tool declarations | Function declarations with name, description and parameter schema, matching inference-time tools |
| `parallel_tool_calls` | boolean | Optional | Retained by tool-call trace examples; model support must be checked separately |

### Role-specific content union

- **system** and **tool**: text string. A tool reply also needs `tool_call_id`.
- **user**: text string or a nonempty array of text/image parts. Text parts are
  `{"type":"text","text":"..."}`; images are
  `{"type":"image_url","image_url":{"url":"...","detail":"auto"}}`.
  `detail` is optional (`auto`, `low`, or `high`). Image URL strings may be public
  URLs or supported base64 data URIs. Local checks do not fetch them or prove
  accessibility, image validity, moderation acceptance or permission to use them.
- **assistant (text target)**: text string or a nonempty array of text parts,
  as used by the bundled ChartQA examples. Assistant image output is **not**
  supported. A text-part array does not imply that all models accept that wire
  form: perform a separately versioned conversion to text if a model requires it.
- **assistant (function-call target)**: `content` may be omitted or `null` when
  there is a nonempty, valid `tool_calls` array. An assistant with neither text
  nor tool calls is not a training target. Null user/system/tool content is invalid.

A record may end in an assistant tool call: it teaches that call and need not
contain a fabricated tool result. If a tool reply is included, its ID must match
an outstanding call, and replies must arrive before the next conversation turn.
[../dataset_integrity.py](../dataset_integrity.py) exposes `validate_sft_row` for
this local sample union; service eligibility and model validation remain separate.

## Role Descriptions

- **system**: Sets the behavior, tone, or context for the assistant (optional but recommended)
- **user**: The input or question from the end user
- **assistant**: The desired response the model should learn to produce
- **tool**: A recorded result of a preceding assistant function call, not an assistant target

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
          "id": "call_stock_1",
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

### Multimodal Example

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "What color is the largest bar?"},
      {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "Blue"}]}
  ]
}
```

The URL is illustrative; no image retrieval or validation is implied. Vision
training requires a vision-fine-tuning-capable model/version/region, not merely
a text or inference-time vision model.

## Validation Checklist

- [ ] File is valid JSONL (one JSON object per line)
- [ ] Each record has a `messages` array
- [ ] Each message has a supported `role` and role-appropriate content union
- [ ] Omitted/null content occurs only on valid assistant tool-call turns
- [ ] Text/image parts match the chosen model's modality support; no assistant image outputs
- [ ] Included tool replies match preceding call IDs; final assistant calls need no invented replies
- [ ] At least one `user` and one `assistant` message per record
- [ ] No trailing commas in JSON objects

## Datasets in this Folder

| Dataset | Description | Records |
|---------|-------------|---------|
| [Text-GSM8K](./Text-GSM8K/) | Grade school math problems | ~7,500 |
| [Multimodal-chartqa](./Multimodal-chartqa/) | Chart interpretation with images | ~18,000 |
| [Tool-Calling](./Tool-Calling/) | Stock market function calling | ~500 |
