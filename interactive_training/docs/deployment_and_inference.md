# Deploy a checkpoint for inference

After training has saved a checkpoint, the Fireworks deployment helper can
create a Foundry deployment from it. For sampling without a deployment, see
[evaluation and inference](./evaluation-and-inference.md).

## Before you deploy

- Have an existing training `session_...` ID with a saved
  checkpoint and confirm the model is eligible for Fireworks deployment in
  [supported models](./supported_models.md).
- Confirm access to the intended tenant, subscription, Foundry project and
  account, including permission to list checkpoints and create deployments.
- Install the cookbook as described in the [quickstart](./quickstart.md), then
  install its optional deployment dependencies in that same environment:

  ```bash
  python -m pip install -e '.[deployment]'
  ```

- For **Fireworks deployment** obtain Fireworks private preview access for model inferencing;
  request it through [Fireworks preview consent](https://aka.ms/fireworks-consent).
  Interactive training access alone does not grant deployment access.

The helper uses `DefaultAzureCredential` for both the project and Azure Resource
Manager. Sign in to the intended Azure Subscription before running it using
Azure CLI:

```shell
az login
az account set --subscription <SUBSCRIPTION_ID>
```

## Create a deployment with the script

From the `fine-tuning/interactive_training` directory, replace the placeholders
with the **Foundry project ARM ID**, and your existing training session ID which will be of the form: `session_<id>`.
The ARM ID is not the project endpoint URL.

```shell
python scripts/deploy_to_foundry.py "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.CognitiveServices/accounts/<ACCOUNT_NAME>/projects/<PROJECT_NAME>" "<SESSION_ID>" --target Fireworks --sku-capacity 50
```

`--target Fireworks` and `--sku-capacity 50` are the script defaults; choose
capacity based on your approved quota and serving needs. The script reports
`Deployment succeeded: <name>` only once provisioning succeeds. It reuses an
existing deployment only if its selected checkpoint and capacity match; it
does not overwrite one with different settings. A timeout does **not** mean
the remote deployment was cancelled: inspect the Foundry account before
retrying. Record the deployment name from the result for inference and cleanup.

## Create a deployment from Python

Run this code in the installed cookbook environment. Unlike the CLI, the
function requires both the target and capacity arguments:

```python
from interactive_training.deployment import DeploymentTarget, create_deployment

project_id = (
    "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/"
    "providers/Microsoft.CognitiveServices/accounts/<ACCOUNT_NAME>/projects/<PROJECT_NAME>"
)
session_id = "<SESSION_ID>"  # will be of the form: session_<id>

deployment = create_deployment(
    project_id,
    session_id,
    DeploymentTarget.FIREWORKS,
    sku_capacity=50,
)
print("Deployment succeeded:", deployment["name"])
```

`create_deployment()` returns the provisioned ARM deployment object before waiting for the deployment to succeed. If no
checkpoint exists, access fails, or an existing deployment has incompatible
settings, it raises an error rather than silently creating another resource.
Keep the deployment name and the chosen checkpoint distinct from the training
session ID.

## Use the deployment for Inference

Use the resulting **deployment name** (not the `session_...` ID or training
checkpoint path) in inferencing requests.

The following Python example uses `AIProjectClient` with Microsoft Entra ID
authentication to obtain an OpenAI client configured for your project.
Replace the project endpoint placeholders and `<DEPLOYMENT_NAME>` with the name returned by the deployment helper.

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
project_client = AIProjectClient(
    endpoint="https://<ACCOUNT_NAME>.services.ai.azure.com/api/projects/<PROJECT_NAME>",
    credential=credential,
)
client = project_client.get_openai_client()

response = client.chat.completions.create(
    model="<DEPLOYMENT_NAME>",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain fine-tuning in one sentence."},
    ],
)
print(response.choices[0].message.content)
```

See [Foundry deployment inference](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/chatgpt) for more details.


## Delete the deployment

When you no longer need the deployment, confirm that you own it and explicitly
delete it using its name:

```shell
python scripts/deploy_to_foundry.py "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.CognitiveServices/accounts/<ACCOUNT_NAME>/projects/<PROJECT_NAME>" --delete "<DEPLOYMENT_NAME>"
```

Deletion waits for the deployment to disappear. Closing a local client or
unloading a training session does not delete a serving deployment.
