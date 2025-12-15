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

## Bootstrapping the Azure Function

1. **Login to Azure:**
   ```bash
   az login
   ```

2. **Create a resource group (if needed):**
   ```bash
   az group create --name <your-resource-group> --location <location>
   ```

3. **Update the `main.parameters.json` file** with your values:
   ```json
   {
     "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
     "contentVersion": "1.0.0.0",
     "parameters": {
       "functionAppName": {
         "value": "my-function-app"
       },
       "cosmosDbAccountName": {
         "value": "my-cosmos-account"
       }
     }
   }
   ```

4. **Deploy:**
   ```bash
   az deployment group create \
     --resource-group <your-resource-group> \
     --template-file main.bicep \
     --parameters main.parameters.json
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