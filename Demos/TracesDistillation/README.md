# Traces → SFT Distillation

This demo fine-tunes a small student model (`gpt-4.1-nano`) to mimic a larger hosted agent's tool-using behavior, using **real production traces** from your Foundry agent as training data. No labels required.

## What it shows

1. **Pull** real conversation traces from a deployed Foundry agent via the `foundry-traces` recipe of the Data Generation API
2. **Transform** the raw export into Azure-FT-ready JSONL (5 fixes the Foundry export currently requires for tool-using agents — all applied inline in the notebook)
3. **Score** the base student on a held-out test set using **structural tool-call comparison**
4. **Submit** one fine-tuning job (winning hyperparameters: 3 epochs, lr=1.0)
5. **Monitor** training to completion
6. **Deploy** the fine-tuned model
7. **Evaluate** it on the same test set and report the lift

Evaluation is driven by the **Foundry evaluations SDK** (`azure-ai-evaluation`) with a custom tool-call structural evaluator.

## Result on the included Zava retail agent

Pulling 720 hours of real traces from a deployed Zava-style Post-Purchase Resolution Desk agent (gpt-4.1-mini behind it, ~100 conversations across 5 sessions) and distilling into `gpt-4.1-nano`:

| Model | Combined | Pass Rate | Lift |
|-------|----------|-----------|------|
| Baseline `gpt-4.1-nano` | 7.38 | 60% | — |
| Fine-tuned `gpt-4.1-nano` (3ep, lr=1.0) | **8.60** | **100%** | **+16.5%** |

The `gpt-4.1-nano` student, fine-tuned on traces from a `gpt-4.1-mini` teacher agent, matches the teacher's tool selection on every test row — at ~10× lower cost per token.

## Prerequisites

- An Azure AI Foundry project with a **deployed hosted agent** that has historical traces in App Insights. If you don't have one yet:
  - Use any existing agent (any hosted Foundry agent emits traces automatically)
  - Or run `generate_traces.py` against a prompt agent with local function tools to populate complete multi-turn trace history
- A project-level **Application Insights connection**. Trace data generation fails with `DataGenerationJobInvalidAppInsightsSetup` when the project has no App Insights connection.
- One **student** model deployment that supports fine-tuning (e.g. `gpt-4.1-nano`, `gpt-4.1-mini`)
- Azure CLI (`az login`) for secretless authentication and deployment
- Python 3.12 with:

```bash
pip install openai>=2.0 azure-ai-projects>=2.2.0 azure-identity>=1.21 azure-ai-evaluation>=1.0
```

## Files in this folder

| File | Purpose |
|------|---------|
| `notebook.ipynb` | End-to-end runnable walkthrough — **fully self-contained**, no external scripts required |
| `generate_traces.py` | Generates complete multi-turn function-calling conversations through a hosted prompt agent without Search or Functions resources |
| `fixtures/push_prompts.py` | Optional standalone script that pushes diverse retail prompts through any hosted agent (use this before the notebook if your agent has no trace history yet) |
| `fixtures/zava_system_prompt.md` | Sample system prompt for the Zava resolution-desk agent (replace with your own) |
| `fixtures/zava_tools.json` | Sample tool catalog (OpenAI chat-completions format) — replace with your own |

## Run it

```bash
set AZURE_AI_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
set AZURE_AI_AGENT_NAME=<your-hosted-agent>
set AZURE_AI_AGENT_VERSION=<version>
set AZURE_SUBSCRIPTION_ID=<subscription-id>
set AZURE_RESOURCE_GROUP=<resource-group>
set AZURE_AI_ACCOUNT_NAME=<resource>

# (Optional) populate complete function-calling trace history first:
python generate_traces.py ^
    --project-endpoint %AZURE_AI_PROJECT_ENDPOINT% ^
    --agent-name <your-hosted-agent> ^
    --model gpt-4.1-mini ^
    --conversations 40

# Wait ~90 seconds for traces to land in App Insights, then:
jupyter notebook notebook.ipynb
```

Full run is ~30–50 minutes depending on FT queue depth.

The notebook uses `AzureCliCredential` through `AIProjectClient.get_openai_client()` and does not require or read an API key. It enumerates hosted agents before trace acquisition and records a prerequisite block when the project has no agent. Fine-tuning is submitted with `trainingType=GlobalStandard`.

## Why the transform step exists

The Foundry traces export currently emits chat JSONL with several issues that Azure FT preprocessing rejects:

1. **Overlapping snapshots** — Each LangGraph node invocation gets its own span; the worker stitches them into one messages list, so one row contains N overlapping snapshots of the same conversation
2. **Fragment rows** — When the customer simulator ends right after the agent asks a clarifying question, you get 2-msg rows with no assistant tool_calls (no SFT signal)
3. **content="null"** — Assistant tool-call rows have `content` as the literal string `"null"` instead of JSON null. Azure FT rejects rows where content is present alongside tool_calls — the field must be **omitted** entirely
4. **Consecutive assistant tool_call turns** — Sometimes two adjacent assistant spans each issue one tool call. Azure FT requires tool replies immediately after each assistant turn; the fix is to merge consecutive `asst(tc)` turns into one assistant message with parallel tool_calls
5. **Missing system prompt + tools array** — The trace export does not currently emit the system message or the tools array at the row level; FT preprocessing for tool-using models needs both

The notebook's 5 inline transform functions apply all five fixes. These should improve as Foundry's export matures — the notebook's transforms are idempotent so it's safe to keep them either way.

## Bring your own agent

Replace `AGENT_NAME` / `AGENT_VERSION` and the two fixture files with your agent's system prompt + tool catalog.
