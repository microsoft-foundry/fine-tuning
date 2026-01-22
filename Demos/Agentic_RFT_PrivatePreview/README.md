# Agentic Reinforcement fine-tuning - Tool Calling & Endpoint Graders
This folder explains how to use two features: endpoint graders and tool calling in [Reinforcement fine-tuning](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning?view=foundry). It includes sample notebooks for each feature, and below we describe the general use of these features. Together, they enable you to train reasoning models like GPT-5 for agentic scenarios.

**Important**: 

- This folder covers documentation and sample code for features available as part of the private preview for Reinforcement Fine Tuning with GPT-5: _**_endpoint graders_**_ and _tool calls during chain-of-thought reasoning_.
- If you do not have access, and would like to request it, fill out this [form](https://forms.office.com/r/UAhqJudDZe).


## Private Preview Expectations & Availability
Private preview features do not have Service Level Agreements (SLAs). Certain features might not be supported or might have constrained capabilities. For more information, see [Supplemental Terms of Use for Microsoft Azure Previews](https://azure.microsoft.com/support/legal/preview-supplemental-terms/).

- **Regional availability**: Reinforcement Training and inferencing is available in:
    - o4-mini : East US2 , Sweden Central  
    - gpt 5 : North Central US, Sweden Central
- **Capabilities**: Only **text-to-text** fine tuning is available. Fine tuning with images or audio is not supported. 
- **Reliability**: o4-mini RFT is GA, while GPT-5 is private preview. Tool calling and endpoint graders are both private preview only features. We will try to fix any issues with private preview features as they arise, but we do not have any SLAs.
- **API Stability**: Tool calling and endpoint graders are part of a preview API and subject to change. 
- **Pricing**: 
    - o4-mini training prices are published on the [Azure OpenAI Pricing Page](https://azure.microsoft.com/en-us/pricing/details/azure-openai/#pricing)
    - GPT-5 training, deployment, and inferencing is free of charge during the private preview. Model grader calls will be billed at the same rate as base model inferencing. We will update customers when billing meters are enabled for GPT-5, and prices will be published to the pricing page.
- **Pause and resume**: user triggered pause and resume does not work for GPT-5 RFT; we cannot guarantee that training runs will produce checkpoints to enable job resumption. Auto-pause (when jobs exceed $5k in charges) has also been disabled for GPT-5 private preview.
- **Responsible use**: All uses of fine tuning must adhere to the [Microsoft Enterprise AI Services Code of Conduct](https://learn.microsoft.com/en-us/legal/ai-code-of-conduct) 

**Support**: If you run into any issues, please contact aliciaframe@microsoft.com, dave.voutila@microsoft.com and keli19@microsoft.com


## Tool Calling

This feature enables users to provide tool definitions that the model can incorporate during chain of thought reasoning. During training, the model can learn how to use custom tools to provide accurate responses. It can invoke multiple tools as needed throughout a rollout, seamlessly switching between independent reasoning and tool calls.

When the model decides to call your tool, it will issue a function call that follows the Responses API function calling format. We will call your `server_url` and send the headers that you specify. 

See below for documentation, or skip straight to sample code in the [RFT_ToolCall Demo](./RFT_ToolCall/)

### To specify tools in the API or SDK

Add an extra `tools` parameter defining the available tools:

```python
job = client.fine_tuning.jobs.create(
  model= "gpt-5-2025-08-07",# or "o4-mini-2025-04-16"
  training_file=train.id,
  validation_file=valid.id,
  suffix="tc-mg-no-eg",
  method={
    "type": "reinforcement",
    "reinforcement": {
      "grader": grader,          
      "tools": tools,            # <-- tool calling 
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
### To specify tools in the UI
In the NextGen Foundry UI, there is a box to paste your tool call definitions. 
### Formatting for tool definitions
Use this structure within your API call, or the field in the NextGen UI, to specify your tools.

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

### Additional Metrics

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

## Endpoint Graders

We support a grader type called `endpoint` for RFT which will allow you to register your own REST endpoint to define custom grading logic, giving you maximum flexibility for advanced and domain-specific use cases. To go straight to sample code & a demo, see  [Endpoint Grader sample](./RFT_EndpointGrader/)

### To use an endpoint grader in the API or SDK

Specify `endpoint` within the grader parameter

```python
job = client.fine_tuning.jobs.create(
  model= "gpt-5-2025-08-07",# or "o4-mini-2025-04-16"
  training_file=train.id,
  validation_file=valid.id,
  suffix="tc-mg-no-eg",
  method={
    "type": "reinforcement",
    "reinforcement": {
      "grader": grader,    # <-- specify your endpoint grader here  
      "tools": tools,            
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
### Specifying an Endpoint grader in the UI
Specify the grader in the grader definition box in NextGen, just like any other grader.

### Structure for specifying the endpoint grader

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

## Samples
For worked examples of using these features, see these demos in our repo:

- [Basic RFT sample - CountDown](../RFT_Countdown/)
- [Endpoint Grader sample](./RFT_EndpointGrader/)
- [Simple Tool calling in RFT](./RFT_ToolCall/)
- [An E2E Agentic RFT sample - Ignite Zava Demo](../ZavaRetailAgent/)