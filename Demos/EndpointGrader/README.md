# Sample Endpoint Grader for use with Countdown Dataset
This demo takes the [Countdown][1] dataset, moves the grader into a remote
Azure Function endpoint, and shows how to use the endpoint grader for Eval
and for RFT jobs.

## Prerequisites
- Azure CLI installed and authenticated
- Python 3.13
- Azure Functions Core Tools v4
- An Azure subscription
- A resource group created

If using Windows, you can use `winget` to install most of the above:

```ps

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
func azure functionapp publish countdown-endpoint-demo --python
```

### 6. Test the function

Without a key, you should get an HTTP 401 response when POSTing something:

```bash
curl -i -d '{}' https://<your-function-app-name>.azurewebsites.net/api/grader
```

```powershell
iwr -Method POST `
   -Headers @{"Content-Type" = "application/json"} `
   -Uri "https://<your-function-app-name>.azurewebsites.net/api/grader"
```

The following example will fetch the key and use it with one of the test files:

```bash
curl -i -d "@test.json" \
   -H "X-Functions-Key: $(az functionapp keys list --resource-group <your-resource-group> \
      --name '<your-function-app-name>' --query 'functionKeys.default' --output tsv)" \
   https://<your-function-app-name>.azurewebsites.net/api/grader
```

```powershell
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

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt \
   pip install -r requirements-demo.txt
   ```

2. **Create and configure a .env file**
   Make sure it's got at least:
   - `AZURE_OPENAI_API_KEY` -- no Entra id used yet
   - `AZURE_OPENAI_ENDPOINT` -- full endpoint like https://amf-finetunes-in-sweden.openai.azure.com/openai/v1/
   - `AZURE_FUNCTIONS_ENDPOINT` -- full path to Azure Function like https://countdown-endpoint-fn.azurewebsites.net/api/grader
   - `X_FUNCTIONS_KEY` -- authentication key to the Azure Function


## Local Testing
The grader logic is in [grader.py](./grader.py). You can test it in two ways:

1. **Direct invocation:**
   ```bash
   python grader.py
   ```

2. **As a local Azure Function:**
   In one shell:
   ```bash
   func start
   ```

   In another:
   ```bash
   curl -s -d @test.json "http://localhost:7071/api/grader"
   ```

[1]: https://github.com/azure-ai-foundry/fine-tuning/tree/main/Demos/RFT_Countdown