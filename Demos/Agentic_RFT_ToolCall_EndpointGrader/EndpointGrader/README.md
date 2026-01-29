# Sample Endpoint Grader for use with Countdown Dataset
This demo takes the [Countdown][1] dataset, moves the grader into a remote
Azure Function endpoint, and shows how to use the endpoint grader for Eval
and for RFT jobs.

## Prerequisites
- Azure CLI installed and authenticated
- Python 3.13 (or newer)
- Azure Functions Core Tools v4
- An Azure subscription
- A resource group created

If using Windows, you can use `winget` to install most of the above:

```powershell
winget install --id Python.Python.3.13
winget install --id Microsoft.AzureCLI
winget install --id Microsoft.AzureFunctionsCoreTools
```

## Bootstrapping the Azure Function

### 1. Login to Azure
```bash
az login
```

### 2. Create a resource group (if needed)
```bash
az group create --name <your-resource-group> --location <location>
```

### 3. Update `main.parameters.json`
Replace `<your-function-app-name>` with your own unique function app name.

```json
{
   "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
   "contentVersion": "1.0.0.0",
   "parameters": {
      "functionAppName": {
      "value": "<your-function-app-name>"
      }
   }
}
```

### 4. Deploy resources
```bash
az deployment group create \
   --resource-group <your-resource-group> \
   --template-file main.bicep \
   --parameters main.parameters.json
```

### 5. Publish function
```bash
func azure functionapp publish <your-function-app-name> --python
```

### 6. Test the function

Without a key, you should get an HTTP 401 response when POSTing something:

```bash
# via sh or bash, etc.
curl -i -d '{}' https://<your-function-app-name>.azurewebsites.net/api/grader
```

```powershell
# via powershell
iwr -Method POST `
   -Headers @{"Content-Type" = "application/json"} `
   -Uri "https://<your-function-app-name>.azurewebsites.net/api/grader"
```

The following example will fetch the key and use it with one of the test files:

```bash
# via sh or bash, etc.
curl -i -d "@test.json" \
   -H "X-Functions-Key: $(az functionapp keys list --resource-group <your-resource-group> \
      --name '<your-function-app-name>' --query 'functionKeys.default' --output tsv)" \
   https://<your-function-app-name>.azurewebsites.net/api/grader
```

```powershell
# via powershell
iwr `
   -Uri "https://<your-function-app-name>.azurewebsites.net/api/grader" `
   -Method POST `
   -Headers @{ `
      "X-FUNCTIONS-KEY" = `
         $(az functionapp keys list --resource-group "<your-resource-group>" `
         --name "<your-function-app-name>" --query "functionKeys.default" `
         --output tsv); `
      "Content-Type" = "application/json" `
   } `
   -Body (Get-Content "test.json" -Raw)
```


## Running the Notebook

### 1. Install dependencies
```bash
pip install -r requirements.txt \
pip install -r requirements-demo.txt
```

### 2. Create and configure a `.env` file
Make sure it's got at least:

- `AZURE_OPENAI_API_KEY` -- no Entra id used yet
- `AZURE_OPENAI_ENDPOINT` -- full endpoint like https://your-foundry-resource.openai.azure.com/openai/v1/
- `AZURE_FUNCTIONS_ENDPOINT` -- full path to Azure Function like https://your-function-app-name.azurewebsites.net/api/grader
- `X_FUNCTIONS_KEY` -- authentication key to the Azure Function


## Local Testing
The grader logic is in [grader.py](./grader.py). You can test it in two ways:

### Direct invocation
The [grader.py](./grader.py) file includes a `__main__` entrypoint, so you can
invoke it directly from the command line:

```bash
python grader.py
```
```
grading with:
        sample: {'output_json': {'expression': '(72 + 22) / (88 - 86)', 'result': 47}}
        item: {'mesages': [{'role': 'developer', 'content': 'You are an expert in arithmetic problem solving...'}, {'role': 'user', 'content': 'target: 47\nnumbers: [86, 22, 88, 72]'}], 'target': 47, 'nums': [86, 22, 88, 72]}
score: 5
```

### As a local Azure Function
In one shell, start a local Azure Function service:

```bash
func start
```

In another shell, POST it some JSON, but _without_ the authentication key:

```bash
curl -s -d @test.json "http://localhost:7071/api/grader"
```

[1]: https://github.com/azure-ai-foundry/fine-tuning/tree/main/Demos/RFT_Countdown