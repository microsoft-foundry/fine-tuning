# Authentication

The cookbook recipes talk to your Azure AI Fine-Tuning Sessions endpoint. They
pick a credential mode automatically based on the endpoint and environment.

## Prerequisite — an Azure AI Foundry resource

Use an Azure AI Foundry resource **and a project under that resource** in your Azure subscription. You can reuse existing ones or create them if authorized. The resource/account key and the **project endpoint** are related but distinct: an account-level endpoint alone is not the recipe's project URL.

> [!IMPORTANT]
> The Foundry resource **must** be in one of the regions listed in [supported_models.md](./supported_models.md#available-regions).

### Confirm access before spending

Ask the project administrator to verify these separately for the intended
tenant, subscription, resource, project, model, and region:

| Gate | Evidence to obtain | Not a substitute |
|---|---|---|
| Authentication mode | The expected identity or an approved resource-key mechanism | A different identity being signed into the CLI |
| Authorization | Applicable role permissions and scope for session creation, training, sampling, checkpoint access, and cleanup | A generic Reader grant or successful token acquisition |
| Model and region | Current interactive eligibility for the exact base model and project region | A renderer existing in the cookbook or a historical benchmark |
| Capacity and quota | The project's applicable training tier and model capacity/quota | A supported-region table or an assumption that retries will create capacity |
| Spend and cleanup | Approved operation/budget, stop conditions, and who can unload the owned session | A local step limit or closing the HTTP client |

This preview guide does not specify a universal built-in role, quota amount,
or read-only capacity check. Those must be confirmed for the actual project;
do not grant broad permissions or create a paid session merely to discover
whether access works. Keep confirmation and identifiers in private operational
records, not in a public issue.

### Using an existing resource

If you already have a Foundry resource in a supported region:

- Find its API key under the resource's **Keys and Endpoint** page if you use key authentication. Follow your organization's key-access policy; never share the key in an issue or notebook output.
- Select the intended project in the Foundry portal and copy its project endpoint from the project details/overview. Labels can vary by portal version. The expected shape is `https://<account>.services.ai.azure.com/api/projects/<project-name>`.

Do not use an ARM resource ID, an inference deployment URL, or just
`https://<account>.services.ai.azure.com/`. Confirm that the key or identity can
access **this project**, and that the selected model is supported in its region.
`az login` and model/region availability alone do not prove authorization or
capacity. Ask the project administrator for the applicable permissions rather
than granting broad roles or changing policy to work around a failed request.

Then skip to [Option 1](#option-1--api-key-easiest-for-local-runs) or [Option 2](#option-2--azure-cli-login-fallback) below.

### Creating a new resource

1. In the Azure portal, **Create a resource** → search for **Azure AI Foundry** → **Create**.
2. Pick a subscription and resource group.
3. **Region:** select one of the regions in the [supported region list](./supported_models.md).
4. Finish the wizard and wait for deployment to complete.
5. Create or select the project under that resource; copy its project endpoint.
6. If using a key, retrieve it through the resource's **Keys and Endpoint** page.
7. Confirm project authorization, selected-model capacity, and
    budget before creating a training session. Resource creation may incur costs
    and is separate from the local cookbook installation.

## Option 1 — API key (easiest for local runs)

Set `AZURE_AI_API_KEY` in your shell before launching the recipe:

```bash
export AZURE_AI_API_KEY="<your-azure-ai-api-key>"
```

In PowerShell the equivalent is `$env:AZURE_AI_API_KEY = "<your-azure-ai-api-key>"`.
The placeholders above are not credentials. Prefer injecting the value through
an approved secret mechanism; literal assignments may remain in shell history.
Do not put a real key into recipe arguments or a checked-in file.

When this variable is present, the recipe uses `AzureKeyCredential` and logs:

```
Using AzureKeyCredential (AZURE_AI_API_KEY).
```

## Option 2 — Azure CLI login (fallback)

If `AZURE_AI_API_KEY` is unset, the Azure launchers use `DefaultAzureCredential`,
but their managed-identity settings differ:

- **Math RL** uses `DefaultAzureCredential()` with managed identity enabled, so
    AML compute jobs can use an attached user-assigned managed identity (UAMI).
    Set `AZURE_CLIENT_ID` to select a specific UAMI. Local runs can fall back to
    Azure CLI credentials through the normal credential chain.
- **Code RL, tool RL, search-tool RL, and tool-NER RL** explicitly exclude
    managed identity. They still use the remaining `DefaultAzureCredential`
    chain, not an Azure-CLI-only credential: configured environment credentials
    and other supported developer credentials can also be selected.
- Other launchers have their own credential setup; consult the recipe before
    assuming managed identity is excluded.

For local use with Azure CLI:

```bash
az login
az account set --subscription "<your-subscription>"
```

When Math RL takes this path, it logs:

```
Using DefaultAzureCredential (MI → CLI fallback; pin AZURE_CLIENT_ID for a specific UAMI).
```

> [!WARNING]
> For Azure CLI authentication, make sure you've logged into the right **tenant** and selected the right **subscription** with `az account set` before launching training. If another credential is configured earlier in the chain, the CLI account selection does not override it. Authentication failures surface when the SDK acquires a token or creates the session.

## Local development endpoint

Math RL does **not** automatically substitute a dummy key for a local HTTP
endpoint. Without `AZURE_AI_API_KEY`, it still uses `DefaultAzureCredential()`;
Azure Core rejects bearer-token authentication over HTTP. Use HTTPS for real
service endpoints. For a local test server that explicitly accepts a test key,
configure a non-secret development key yourself; never send a real Azure API
key or access token over plain HTTP. Local-endpoint handling is recipe-specific.

## `project_endpoint`

Both paths additionally require the project endpoint — pass it as a CLI argument to the recipe:

```bash
python -m interactive_training.recipes.math_rl.train_azure \
    project_endpoint="https://<your-project>.services.ai.azure.com/api/projects/<project-name>" \
    ...
```

## Verbose HTTP

To debug auth/wire issues, pass `verbose_http=true` to the recipe; it enables the SDK's request/response logging. Treat verbose logs and notebook outputs as potentially sensitive and review them before sharing. Never commit API keys or access tokens.

## Notebook endpoint configuration

Shared notebooks accept `AZURE_AI_PROJECT_ENDPOINT`; `FOUNDRY_PROJECT_ENDPOINT`
and `INTERACTIVE_POST_TRAINING_PROJECT_ENDPOINT` are compatibility aliases. Recipe CLIs still take
the explicit `project_endpoint=...` argument unless their own help documents an
environment default. Setting a notebook variable does not configure every CLI.
