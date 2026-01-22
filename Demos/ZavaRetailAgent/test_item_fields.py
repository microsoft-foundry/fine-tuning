"""Quick test to see what fields are available in output items"""

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

resource_name = "omi-ignite-demo-resource"
eval_id = "eval_691c31f82bf0819199e55210bf0595a0"

# Create client
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=f"https://{resource_name}.openai.azure.com",
    azure_ad_token_provider=token_provider,
    api_version="2025-04-01-preview"
)

# Get first run
runs = list(client.evals.runs.list(eval_id=eval_id))
first_run = runs[0]

print(f"Run: {first_run.name}")
print(f"Run ID: {first_run.id}")

# Get first output item
output_items = client.evals.runs.output_items.list(
    eval_id=eval_id,
    run_id=first_run.id
)

first_item = next(iter(output_items))

print(f"\n=== Output Item Structure ===")
print(f"ID: {first_item.id}")
print(f"Status: {first_item.status}")

print(f"\n=== Sample Object ===")
if hasattr(first_item, 'sample') and first_item.sample:
    sample = first_item.sample
    print(f"Sample attributes: {dir(sample)}")
    for attr in ['messages', 'tools', 'parallel_tool_calls']:
        if hasattr(sample, attr):
            val = getattr(sample, attr)
            print(f"\n{attr}: {type(val).__name__}")
            if val:
                print(f"  {val}")

print(f"\n=== Datasource Item ===")
if hasattr(first_item, 'datasource_item') and first_item.datasource_item:
    ds = first_item.datasource_item
    print(f"Keys: {ds.keys() if isinstance(ds, dict) else 'Not a dict'}")
    if isinstance(ds, dict):
        for key in ds.keys():
            print(f"\n{key}:")
            print(f"  {ds[key]}")

print(f"\n=== Results ===")
if hasattr(first_item, 'results') and first_item.results:
    for result in first_item.results:
        print(f"Result attributes: {dir(result)}")
        for attr in ['score', 'name', 'pass_', 'trace', 'output']:
            if hasattr(result, attr):
                val = getattr(result, attr)
                print(f"  {attr}: {val}")
