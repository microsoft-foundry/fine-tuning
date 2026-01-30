# Getting Started with Fine-Tuning in AI Foundry

This guide walks you through everything you need to run your first fine-tuning demo.

## Prerequisites Checklist

### Azure Requirements

- [ ] **Azure Subscription** with billing enabled
- [ ] **Azure AI Foundry workspace** created at [ai.azure.com](https://ai.azure.com)
- [ ] **Azure OpenAI resource** provisioned in a [supported region](https://learn.microsoft.com/azure/ai-services/openai/concepts/models)
- [ ] **Quota available** for your target model (check in Azure Portal → Azure OpenAI → Quotas)

### Role Assignments

Ensure you have these roles on your Azure AI Foundry project:

| Role | Purpose |
|------|---------|
| **Azure AI Developer** | Create and manage fine-tuning jobs |
| **Storage Blob Data Contributor** | Upload training data |
| **Cognitive Services OpenAI User** | Deploy and use models |

### Local Environment

- [ ] **Python 3.9+** installed ([download](https://www.python.org/downloads/))
- [ ] **pip** package manager (included with Python)
- [ ] **Jupyter Notebook** or **VS Code** with Jupyter extension
- [ ] **Git** for cloning this repository

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/Azure/fine-tuning.git
cd fine-tuning
```

---

## Step 2: Choose Your First Demo

We recommend starting with **Supervised Fine-Tuning (SFT)** as it covers the most common use cases.

| If you want to... | Start with this demo |
|-------------------|---------------------|
| Learn the basics | [SFT_CNN_DailyMail](Demos/SFT_CNN_DailyMail/) |
| Work with medical data | [SFT_PubMed_Summarization](Demos/SFT_PubMed_Summarization/) |
| Try preference optimization | [DPO_Intel_Orca](Demos/DPO_Intel_Orca/) |
| Explore reinforcement learning | [RFT_Countdown](Demos/RFT_Countdown/) |
| Fine-tune vision models | [Image_Breed_Classification_FT](Demos/Image_Breed_Classification_FT/) |

---

## Step 3: Set Up the Demo Environment

Navigate to your chosen demo folder:

```bash
cd Demos/SFT_CNN_DailyMail  # or your chosen demo
```

### Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

1. Copy the template:
   ```bash
   cp .env.template .env
   ```

2. Edit `.env` with your Azure details:
   ```
   MICROSOFT_FOUNDRY_PROJECT_ENDPOINT=https://your-project.api.azureml.ms
   AZURE_SUBSCRIPTION_ID=your-subscription-id
   AZURE_RESOURCE_GROUP=your-resource-group
   AZURE_AOAI_ACCOUNT=your-openai-account
   ```

3. Find these values in [Azure Portal](https://portal.azure.com) or [AI Foundry](https://ai.azure.com)

---

## Step 4: Run the Notebook

1. Open the notebook in VS Code or Jupyter:
   ```bash
   jupyter notebook  # Opens in browser
   # or
   code .  # Opens VS Code
   ```

2. Select your Python environment (the `.venv` you created)

3. Run cells sequentially from top to bottom

4. Monitor your fine-tuning job in [AI Foundry](https://ai.azure.com) → Jobs

---

## Step 5: Understand the Data Format

Each fine-tuning technique requires a specific data format. See the schema documentation:

- **SFT**: [Supervised_Fine_Tuning/SCHEMA.md](Sample_Datasets/Supervised_Fine_Tuning/SCHEMA.md)
- **DPO**: [Direct_Preference_Optimization/SCHEMA.md](Sample_Datasets/Direct_Preference_Optimization/SCHEMA.md)
- **RFT**: [Reinforcement_Fine_Tuning/SCHEMA.md](Sample_Datasets/Reinforcement_Fine_Tuning/SCHEMA.md)

---

## Common Issues & Solutions

### Authentication Errors

**Error**: `DefaultAzureCredential failed to retrieve a token`

**Solution**: 
```bash
az login
az account set --subscription "your-subscription-id"
```

### Quota Exceeded

**Error**: `Insufficient quota for model deployment`

**Solution**: 
- Request quota increase in Azure Portal → Azure OpenAI → Quotas
- Or try a different region with available capacity

### File Not Ready

**Error**: `File status is not 'ready'`

**Solution**: 
- Wait a few minutes for file processing to complete
- Check file format matches the expected schema
- Ensure file is valid JSONL (one JSON object per line)

### Training Job Failed

**Error**: `Training job failed with status 'Failed'`

**Solution**:
- Check job logs in AI Foundry for specific error
- Verify data format matches schema requirements
- Ensure sufficient quota for training

---

## Cost Considerations

Fine-tuning jobs incur costs based on:
- **Training tokens processed** (input + output)
- **Training duration** (compute hours)
- **Model hosting** (if you deploy the fine-tuned model)

**Tips to minimize costs during learning**:
- Use smaller datasets first (100-500 examples)
- Set `n_epochs` to 1-2 for initial experiments
- Delete fine-tuned models and deployments when done testing

---

## Next Steps

After completing your first demo:

1. **Try a different technique**: If you started with SFT, try [DPO](Demos/DPO_Intel_Orca/) or [RFT](Demos/RFT_Countdown/)
2. **Use your own data**: Format your data following the [schema documentation](Sample_Datasets/)
3. **Explore advanced demos**: [ZavaRetailAgent](Demos/ZavaRetailAgent/) shows a complete agent workflow
4. **Read the docs**: [Azure AI Foundry documentation](https://learn.microsoft.com/azure/ai-studio/)

---

## Getting Help

- **Issues with this repo**: [Open an issue](https://github.com/Azure/fine-tuning/issues)
- **Azure AI Foundry questions**: [Microsoft Q&A](https://learn.microsoft.com/answers/tags/azure-ai-services)
- **Azure OpenAI support**: [Azure Support](https://azure.microsoft.com/support/)
