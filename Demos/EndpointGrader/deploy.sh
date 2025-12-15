#!/bin/bash
# Deploy Azure Functions to Flex Consumption plan

set -e

# Configuration
RESOURCE_GROUP="${1:-dv-zava-agentic-rft-rg}"
FUNCTION_APP_NAME="${2:-countdown-endpoint-fn}"

echo "==================================="
echo "Deploying Azure Functions"
echo "==================================="
echo "Resource Group: $RESOURCE_GROUP"
echo "Function App:   $FUNCTION_APP_NAME"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI is not installed"
    exit 1
fi

# Check if logged in
if ! az account show &> /dev/null; then
    echo "Not logged in to Azure. Running 'az login'..."
    az login
fi

# Option 1: Using func core tools (if available)
if command -v func &> /dev/null; then
    echo "Deploying using Azure Functions Core Tools..."
    func azure functionapp publish "$FUNCTION_APP_NAME" --python
else
    echo "Install the Azure Functions Core Tools please!" >&2
    exit 1
fi

echo ""
echo "==================================="
echo "Deployment completed successfully!"
echo "==================================="
echo ""

# Get Application Insights connection string
echo "Application Insights Connection String:"
az functionapp config appsettings list \
    --name "$FUNCTION_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "[?name=='APPLICATIONINSIGHTS_CONNECTION_STRING'].value" -o tsv

echo ""
echo "Function URLs:"
az functionapp function show \
    --name "$FUNCTION_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --function-name http_trigger \
    --query "invokeUrlTemplate" -o tsv 2>/dev/null || echo "Run 'az functionapp function list' to see all functions"

echo ""
echo "View Application Insights:"
AI_NAME=$(az monitor app-insights component show --resource-group "$RESOURCE_GROUP" --query "[?tags.FunctionApp=='$FUNCTION_APP_NAME'].name" -o tsv 2>/dev/null || \
          az monitor app-insights component list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv 2>/dev/null)
if [ -n "$AI_NAME" ]; then
    echo "https://portal.azure.com/#resource/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/microsoft.insights/components/$AI_NAME"
else
    echo "Application Insights resource name not found automatically"
fi

echo ""
echo "To view logs:"
echo "  az functionapp log tail --name $FUNCTION_APP_NAME --resource-group $RESOURCE_GROUP"
