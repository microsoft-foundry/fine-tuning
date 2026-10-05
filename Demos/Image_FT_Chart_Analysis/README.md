# Fine-tuning Azure OpenAI GPT-4.1 for Chart Analysis

This notebook demonstrates the process of vision fine-tuning of GPT-4.1 leveraging a chart analysis benchmark dataset for visual and logical reasoning. It covers data preparation, fine-tuning, deployment, and evaluation against GPT-4.1's baseline performance.

We are using the **ChartQA** dataset, a benchmark designed for question answering tasks involving chart images. Each entry in the dataset comprises a chart image, an associated question, and the corresponding answer, facilitating the development and evaluation of models that integrate visual and logical reasoning to interpret and analyze information presented in graphical formats.

<img src="qna.png" alt="Frame Samples" width="1000">

__Acknowledgements:__  
This project utilizes the ChartQA dataset introduced by Masry et al. in their paper, *ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning* (Findings of ACL 2022). We acknowledge the authors for providing this valuable resource. For more details, refer to the publication: [ChartQA: ACL 2022](https://aclanthology.org/2022.findings-acl.177).

## Get started

Create and activate a virtual Python environment for running the code.
The following example shows how to create a Conda environment named `vision-ft`:

```bash
conda create -n vision-ft python=3.12
conda activate vision-ft
```

Install the required packages. Navigate to the `Image_FT_Chart_Analysis` folder and execute the following:

```bash
pip install -r requirements.txt
```

__Required Services:__
- An Azure AI Foundry project with:
  - A GPT-4.1 deployment.
  - Fine-tuning access for GPT-4.1.
  - A deployment for the succeeded fine-tuned model before running evaluation.

The notebook uses `azure-ai-projects` 2.x and Microsoft Entra ID. Sign in with
`az login`, copy `.env.template` to `.env`, and set the project endpoint and
deployment names. No API key, Azure OpenAI account endpoint, or manually
constructed service route is required.

The Azure AI Foundry SDK currently exposes project-scoped deployment discovery,
not deployment creation. Create the fine-tuned deployment in the Foundry portal,
then set `CHART_FT_DEPLOYMENT`. You can set `CHART_FT_REUSE_JOB_ID` to the
matching succeeded job when rerunning the notebook without starting another
training job.

Navigate to the vision fine-tuning notebook:

- [Fine-tuning Azure OpenAI GPT-4.1 for Chart Analysis](fine-tune-aoai-gpt4-1-for-chart-analysis.ipynb)

## Troubleshooting

### Common Issues

**Authentication Error**
- Verify `.env` contains the Azure AI Foundry project endpoint
- Run `az login` to refresh Azure credentials
- Ensure your identity has access to the project
- Ensure the project has GPT-4.1 deployed

**Training Job Fails**
- Verify JSONL format matches the expected schema
- Ensure base64 image encoding is correct
- Check that images don't exceed size limits

**Content Filter Triggered**
- Some chart images may trigger content filters
- Use the [official form](https://customervoice.microsoft.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbR7en2Ais5pxKtso_Pz4b1_xUMlBQNkZMR0lFRldORTdVQzQ0TEI5Q1ExOSQlQCN0PWcu) for policy adjustments if needed

**Quota Exceeded**
- Request additional quota in Azure Portal → Azure OpenAI → Quotas
- Try a different Azure region with available capacity
