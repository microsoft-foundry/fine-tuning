# Agentic Reinforcement fine-tuning - Tool Calling & Endpoint Graders

**Important**: 

- This folder covers Private Preview Agentic reinforcement fine tuning with o4-mini/ gpt 5 in Microsoft Foundry/Azure Open AI service. To request access, fill out this [form](https://forms.office.com/r/UAhqJudDZe).
- This preview is provided without a service-level agreement, and we don't recommend it for production workloads. Certain features might not be supported or might have constrained capabilities. For more information, see [Supplemental Terms of Use for Microsoft Azure Previews](https://azure.microsoft.com/support/legal/preview-supplemental-terms/).

## Private Preview Expectations

1. Model support & scope : This private preview offers access to Endpoint graders and Tool calling with o4-mini/ gpt5 reinforcement fine tuning. 
2. Regional availability: Reinforcement Training and inferencing is available in:
    - o4-mini : East US2 , Sweden Central  
    - gpt 5 : North Central US, Sweden Central

3. Capabilities: Only **text-to-text** fine tuning is available. Fine tuning with images or audio is not supported. 
4. Surfaces: The private preview is available only via the AI Foundry and AOAI Studio UIs (for training, deployment, and inferencing) and the AOAI Python SDK & REST API (for deployment and inferencing).
5. Reliability: o4-mini is GA service while gpt 5 is private preview only. We will fix gpt 5 issues under reasonable time frames.
6. API Stability: This is a preview API and subject to change. 
7. Retraining fine-tuned models: It is possible that you may need to re-train your fine-tuned model in the future to be able to continue using it. We cannot guarantee long term support for any fine-tuned models trained and deployed during the private preview. 
8. Content moderation and abuse detection: Content filters (learn more here) and abuse monitoring are in place as described in the linked documentation. Customers will not be approved for Modified Content Filtering or Modified Abuse Monitoring for Azure subscriptions accessing the fine-tuning private preview. 
9. Quotas and limits: Fine tuning is subject to the following limits: 
    - Training: 1 active training run and up to 10 queued jobs
    - Deployments: Up to 5 deployments per region / subscription across all fine-tuned models (contact us if you need additional deployments) 
10. Pricing: Since o4-mini is a GAed service, it will be charge normally. GPT5 RFT training is free during private preview. Customers who use model based graders will be charged for inferencing tokens. Auto-pause at $5k spend is not enabled during the private preview. There is no separate pricing for Tool calling RFT or Endpoint graders.
11. Responsible use: All uses of fine tuning must adhere to the Azure OpenAI Code of Conduct. 

**Support**: If you run into any issues, please contact aliciaframe@microsoft.com, dave.voutila@microsoft.com and keli19@microsoft.com


This folder explains how to use two features: endpoint graders and tool calling in [Reinforcement fine-tuning](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning?view=foundry). It includes sample notebooks for each feature, and below we describe the general use of these features. Together, they enable you to train reasoning models like GPT-5 for agentic scenarios."

## Tool Call

During training, the model learns to incorporate tool usage within its reasoning process. It can invoke multiple tools as needed throughout a rollout, seamlessly switching between independent reasoning and tool calls.

When the model decides to call your tool, it will issue a function call that follows the Responses API function calling format. We will call your `server_url` and send the headers that you specify.

### Specify a list of tools the model needs to use

```python
tools = [ 
{ 
"name": "current_date",

"server_url": "https://rft-endpointfn.azurewebsites.net/api/tools",
 
"headers": { 
   "X-Functions-Key": "<Add bearer token>" } 
}, 


{ "name": "get_by_id",
 
"server_url": "https://rft-endpointfn.azurewebsites.net/api/tools", 

"headers": { 
    "X-Functions-Key": "<Add bearer token>" } 
},

 { "name": "search", 

"server_url": "https://rft-endpoint-fn.azurewebsites.net/api/tools", 

"headers": { 
    "X-Functions-Key": "<Add bearer token>"  } 
},

 { "name": "delete_by_id", 

"server_url": "https://rft-endpoint-fn.azurewebsites.net/api/tools",

 "headers": { 
"X-Functions-Key": "<Add bearer token>" }
 } 

]

```

### What tool call returns

The tool should return in a format identical to a function_call_output in the Responses API format.

```json

{ 

"type": "function_call_output", 

"call_id": "call_12345xyz",

"output": "The numbers in the mathematical operations for countdown puzzle are..", 

"id": "fc_12345xyz" 

}
```


**Note**

- `server_url` and `headers`must be identical to whatever you specified in the job and dataset file.

- X-Functions-Key” is not an actual required header. There are no required headers.

### Create a RFT job with tool calling

RFT job will be defined as:

```python
job = client.fine_tuning.jobs.create(
  model= "gpt-5-2025-08-07",#"o4-mini-2025-04-16", "gpt-5-2025-08-07"
  training_file=train.id,
  validation_file=valid.id,
  suffix="tc-mg-no-eg",
  method={
    "type": "reinforcement",
    "reinforcement": {
      "grader": grader,          # <-- endpoint grader here
      "tools": tools,            # <-- tool calling still here
      "max_episode_steps": 10,
      "hyperparameters": {
        "eval_interval": 3,
        "eval_samples": 5,
        "compute_multiplier": 1.0,
        "reasoning_effort": "medium"
      }
    }
  }
)

```

### Metrics

There are 4 additional training metrics which are generated for RFT jobs with tool calls. You can view these metrics in the **Metrics** tab in the finetune job detail page in Foundry UI.

- Tool calls per roll out : the number of tool calls done to individual tools by the training and the validation dataset during RFT for each of the specific training steps

- Mean Tokens per tool call

- Tool Execution Latency

- Tool Errors Count

### Limitations

- Expected QPS - We recommend that your tool endpoints handle 50 QPS.

- Max input payload size - We will limit the input payload size to 1 MB. 

- Max return payload size - We will limit the return payload size to 1 MB. Anything exceeding this will appear as a 413 Error to the model. 

- Timeouts - The maximum amount of time we will wait for a tool response is 10 minutes. 

- Parallel tool calls - It’s possible for our models to call multiple tools at once during a given rollout. Please ensure that you handle race conditions across your tools. You can set parallel tool calling parameter.
 
- If tools fail:
    
    - If a given tool returns a 4xx without the output field -We will serialize the error payload and display that to the model.
    
    - If a given tool returns a 5xx  - We will retry for at most 3 times before discarding that entire rollout.

## Endpoint Grader

We are adding a new grader type called `endpoint` for RFT which will allow you to register your own REST endpoint to define custom grading logic, giving you maximum flexibility for advanced and domain-specific use cases.

### Specify the endpoint grader

```python
endpoint_grader = { 

"type": "endpoint", # enforced literal 
"name": "some_name" # optional
"url": "https://example.com/grade", # user-specified grading endpoint 
"headers": { 
     "Authorization": "Bearer ${KEY}" 
}, # optional, headers will be forwarded to user endpoint  
"rate_limit": 50, # optional, max requests per second sent to the endpoint 
"pass_threshold": 8 


}
```

The RFT job will be defined like:

```Python
job = client.fine_tuning.jobs.create(
  model="o4-mini-2025-04-16", #"gpt-5-2025-08-07",
  training_file=train.id,
  validation_file=valid.id,
  suffix="tc-tools-endpoint-grader-no-auth",
  method={
    "type": "reinforcement",
    "reinforcement": {
      "grader": endpoint_grader,  # <-- endpoint grader (headers are optional)
      "tools": tools,             # <-- tool calling
      "max_episode_steps": 10,
      "hyperparameters": {
        "eval_interval": 3,
        "eval_samples": 5,
        "compute_multiplier": 1.0,
        "reasoning_effort": "medium"
      }
    }
  }
)


```


**Note**

- `"X-Functions-Key": "**********"` is not an actual required header. There are no required headers.
- Currently only key-based authentication is supported.


### Endpoint format and arguments

The grader must accept POST requests at the specified URL, with the following payload:

```JSON 
{ 

"sample": { ... }, // Model outputs to be graded (dict<string, any>) 

"item": { ... }, // Datapoint and references from the training/eval dataset you provided 

"trace_id": "trace_1a2b3c" 

}

```

Arguments include:

- `sample`: dict[str, Any]  : A chat completions style output 
- `item`: dict[str, Any]  : The entire example 
- `trace_id`: str : A UUID prefixed by trace_ that remains the same for all tool calls across a rollout. 

The endpoint grader should return the graded score in JSON format as below:

```json
{ "score": float }

```

### Limitations

- Estimated throughput: We recommend designing your endpoint to handle 50 requests per second during training, although actual RPS during training may be lower depending on your task.
- Max input payload size: 1 MB 
- Timeout: The maximum amount of time we will wait for a grader is 10 minutes.
- If a call to a grader endpoint fails for any reason, we will retry for at most 3 times before discarding the rollout.
- Auto-pause at $5k spend is not enabled during the private preview
- Users triggered pausing jobs does not currently work for GPT5 RFT and we're working on a fix.

## Samples

- [Basic RFT sample - CountDown](../RFT_Countdown/)
- [Endpoint Grader sample](./EndpointGrader/)
- [Simple Tool calling in RFT](./RFT_ToolCall/)
- [An E2E Agentic RFT sample - Ignite Zava Demo](../ZavaRetailAgent/)
