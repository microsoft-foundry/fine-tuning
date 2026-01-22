# Agentic RFT E2E Sample — Tool Calling + (Optional) Endpoint Grader (No Auth)


**Purpose / Disclaimer (Please read first)**  

- This repository is a **simple, hands-on sample** to help you quickly learn and validate the **end-to-end workflow** for **Reinforcement Fine-Tuning (RFT) with tool calling**.  

    - The **tool endpoint** in this sample is a **dummy/mock tool** (a tiny in-memory “catalog search”) and is **NOT** intended to represent a real production tool or real business logic

    - The **graders** in this sample (multi-grader and endpoint grader) are **demo graders** designed only to validate the wiring and execution path. They are **NOT** meant to be used “as-is” for real evaluation or real reward modeling. 

- For production scenarios, you should implement your own tool endpoints, your own grading logic, and apply appropriate security controls. 

---

## What this sample demonstrates

This repository is an end-to-end (E2E) starter sample for **Agentic Reinforcement Fine-Tuning (RFT)** with:

1) **Tool calling** during training (the model calls your HTTP tool endpoint), and  
2) **Either**:
   - **Multi-grader** (built-in graders like `string_check` + `text_similarity`), **or**
   - **Endpoint grader** hosted by you (this sample demonstrates **endpoint grader without auth headers**).

> ✅ This sample is intended for evaluation/experimentation and customer onboarding reference. It is not production-hardened.

---

## What you will build

You will deploy an **Azure Functions (Python)** app (FastAPI + ASGI) that exposes:

- `POST /tool/search_catalog` — **Tool endpoint** used during RFT rollouts  
- `POST /score` — **Endpoint grader** returning `{ "score": <float> }` (no auth headers)

Then you will run two jobs:

- **Job A:** Tool calling + **Multi-grader** (no endpoint grader)  
- **Job B:** Tool calling + **Endpoint grader** (no auth headers)

---

## Quickstart (TL;DR)

1) Deploy Function App (tool + grader endpoints)  
2) Confirm cloud endpoints work:
   - `POST https://<APP>.azurewebsites.net/tool/search_catalog`
   - `POST https://<APP>.azurewebsites.net/score` (no headers)
3) Upload `train_tool.jsonl` / `valid_tool.jsonl`  
4) Submit RFT job with:
   - `tools: [...]` pointing to `/tool/search_catalog`
   - `grader: multi` **or** `grader: endpoint` pointing to `/score` (no headers)

---

## Repository layout (recommended)

.
├─ function_app/                  # Azure Functions (Python) app
│  ├─ function_app.py             # AsgiFunctionApp + FastAPI routes
│  ├─ host.json                   # routePrefix config
│  ├─ requirements.txt
│  └─ local.settings.json         # local only (do not commit secrets)
├─ data/
│  ├─ train_tool.jsonl
│  └─ valid_tool.jsonl
├─ rft-tool-call.ipynb            # Create RFT job with tool cal with endpoint grader
└─ README.md


## Prerequisites

### Access
- Your Azure subscription must be enabled for **RFT tool calling** and (if using Job B) **endpoint graders**.
- You must have an Azure AI Foundry / Azure OpenAI endpoint + API key for the region where RFT is enabled for your model.

### Local tooling
- Python **3.10+** (3.11 recommended)
- Azure CLI (`az`)
- Azure Functions Core Tools v4 (`func`)
- Optional: Conda/venv

---

## 1) Configure environment variables

Create a `.env` file in your project root (or set env vars in your shell). Example:

**Note** : to use endpoint grader, the API version needs to be `v1`.

```bash
# Azure OpenAI / Foundry
AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
AZURE_OPENAI_API_KEY="<your-key>"
AZURE_OPENAI_API_VERSION="v1"

# Azure Function App
FUNC_APP_NAME="<your-function-app-name>"
RG="<your-resource-group>"
LOC="swedencentral"   # example

# Optional: if your tool endpoint requires a key
FUNC_KEY="<functions-key>"
```

## 2) Azure Functions app (tool + grader)

This sample uses FastAPI hosted via Azure Functions ASGI. Endpoints:

- /tool/search_catalog (tool)
- /score (endpoint grader)

### 2.1 Install dependencies (local)

Install required dependencies:

```bash

pip install -r requirements.txt

```

### 2.2 Run locally

``` bash
cd function_app
func start --port 7072
```

If 7072 is taken, choose another port (e.g. 7073).

### 2.3 Test locally (no auth on /score)

#### (A) Endpoint grader (no headers)

```bash
curl -i -X POST "http://localhost:7072/score" \
  -H "Content-Type: application/json" \
  -d "{\"sample\":{\"output_text\":\"JKT-URB-009\"},\"item\":{\"reference_answer\":\"JKT-URB-009\"},\"trace_id\":\"trace_test\"}"
```

Expected: HTTP/1.1 200 OK and JSON like:

``` bash
{"score": 1.0}
```

#### (B) Tool endpoint (direct args)

```bash

curl -i -X POST "http://localhost:7072/tool/search_catalog" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Urban Puffer\",\"top_k\":1}"

```

Expected: HTTP/1.1 200 OK with a tool output payload.

## 3) Deploy Azure Functions to Azure

### 3.1 One-time: create resources

```bash

az login
az group create -n "$RG" -l "$LOC"

# Storage name must be globally unique
STORAGE="st$RANDOM"rft2
az storage account create -n "$STORAGE" -g "$RG" -l "$LOC" --sku Standard_LRS

az functionapp create -n "$FUNC_APP_NAME" -g "$RG" -s "$STORAGE" \
  --consumption-plan-location "$LOC" --runtime python --functions-version 4

```

### 3.2 Configure app settings (cloud)

To make the endpoint grader anonymous in Azure, set:
```bash

az functionapp config appsettings set -g "$RG" -n "$FUNC_APP_NAME" \
  --settings SCORE_AUTH_DISABLED=true


```

(Optional) If you also want the tool endpoint anonymous:

```bash

az functionapp config appsettings set -g "$RG" -n "$FUNC_APP_NAME" \
  --settings TOOL_AUTH_DISABLED=true

```

Restart for settings to take effect:

```bash
az functionapp restart -g "$RG" -n "$FUNC_APP_NAME"

```

### 3.3 Publish code

From the function_app/ directory:
```bash

cd function_app
func azure functionapp publish "$FUNC_APP_NAME"
```

### 3.4 Validate in Azure (no auth headers)

#### Endpoint grader (no headers)

```bash

curl -i -X POST "https://$FUNC_APP_NAME.azurewebsites.net/score" \
  -H "Content-Type: application/json" \
  -d "{\"sample\":{\"output_text\":\"JKT-URB-009\"},\"item\":{\"reference_answer\":\"JKT-URB-009\"},\"trace_id\":\"trace_test\"}"

```

Expected: HTTP/1.1 200 OK and {"score":...}.

## 4) Contracts: Tool calling + Endpoint grader

These shapes matter. Use them exactly.

### 4.1 Tool call input (what the training service sends)

During training, the tool endpoint may receive a wrapper object like:
```json

{
  "type": "function_call",
  "id": "fc_123",
  "call_id": "call_123",
  "name": "search_catalog",
  "arguments": "{\"query\":\"Urban Puffer\",\"top_k\":3}",
  "trace_id": "trace_abc",
  "item": {}
}

```

### 4.2 Tool output (what your tool must return)

Return a Responses-style function_call_output:

```json

{
  "type": "function_call_output",
  "call_id": "call_123",
  "id": "fc_123",
  "output": "{\"items\":[...]}"
}

```

### 4.3 Endpoint grader request (what the training service sends)

Your /score endpoint receives:

```json

{
  "sample": { "output_text": "..." },
  "item": { "reference_answer": "..." },
  "trace_id": "trace_abc"
}

```

## 5) Dataset format (JSONL)

Your `train_tool.jsonl` and `valid_tool.jsonl` must be JSONL:

- One valid JSON object per line
- Must include messages (array)
- Can include additional fields such as reference_answer
- At least 10 records per file

Example minimal line:

```json
{"messages":[{"role":"user","content":"Return the SKU of the cheapest warm jacket under $200."}],"reference_answer":"JKT-URB-009"}
```


### Tool-calling training

To train tool use, include examples where the model learns to call tools. Ensure:

- The tool name used in training matches the tool name in the job config (search_catalog)
- The job config includes the tool in method.reinforcement.tools


## 6) Run the E2E jobs (Python notebook)

Run celss in `rft-tool-call.ipynb`.

Both flows:

0. Upload training + validation JSONL files (purpose="fine-tune")
2. Wait for files to process
3. Create the RFT job
4. Print job ID and initial status


Ensure `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_API_VERSION` are set before running.


## 7) Monitor job status

You can poll the job status in a loop:

```python

import time
j = client.fine_tuning.jobs.retrieve(job.id)
while j.status not in ("succeeded", "failed", "cancelled"):
    print("status:", j.status)
    time.sleep(30)
    j = client.fine_tuning.jobs.retrieve(job.id)

print("final:", j.status)
print("fine_tuned_model:", getattr(j, "fine_tuned_model", None))

```


Or view the job in Foundry UI.

## Troubleshooting

### 401 Unauthorized calling /score

- Confirm Azure app setting is set:
    - SCORE_AUTH_DISABLED=true

- Restart the Function App after changing settings.

- Ensure App Service Authentication (EasyAuth) is disabled (if enabled, it can force auth before your code runs).

### 404 on /score or /tool/search_catalog

- Check host.json routePrefix:
    - If routePrefix is "", endpoints are /score and /tool/search_catalog
    - If default route prefix is used, endpoints may be under /api/... (e.g., /api/score)


### Port already in use locally

- Use a different port:

```bash

func start --port 7073Show more lines
```


### Tool endpoint returning 401

- If your tool endpoint requires a key, include it in your job config tools[].headers.
- If you want the tool endpoint anonymous, set TOOL_AUTH_DISABLED=true in Azure app settings and restart.


## Limitations

- Endpoint grader does not support auth header for now, and we'll support it soon.
