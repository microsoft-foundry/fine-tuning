# Fine-Tuning GPT-4.1 for Action Recognition in Video Clips

This Solution Accelerator demonstrates how to use Azure OpenAI GPT-4.1 vision fine-tuning to improve the model's performance in detecting human activities in video clips. The project utilizes the [UCF101 - Action Recognition](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) dataset from Kaggle, a comprehensive video dataset featuring 101 distinct human action categories such as "playing guitar," "surfing," and "knitting." It contains 13,320 video clips, each labeled with a single action category.

Below are examples of video frames representing 8 of the 101 classes in the dataset:

<img src="frame-samples.png" alt="Frame Samples" width="1000">

__Acknowledgements:__

- Dataset: https://www.crcv.ucf.edu/research/data-sets/ucf101/
- Citation: https://arxiv.org/abs/1212.0402

This Solution Accelerator provides reusable code to help you apply vision fine-tuning for video analysis in various use cases.

> **Note:** By default, the Fine-Tuning API rejects images containing people and faces for AI safety reasons.
You can apply to opt out of this limitation by submitting a request here: https://customervoice.microsoft.com/Pages/ResponsePage.aspx.
Make sure to select the Modified Content Filtering option for Inferencing and Fine Tuning in question 14.


## Get started

Create and activate a virtual Python environment for running the code.
The following example shows how to create a Conda environment named `video-ft`:

```bash
conda create -n video-ft python=3.12
conda activate video-ft
```

Install the required packages. Navigate to the `Video_FT_Action_Recognition` folder and execute the following:

```bash
pip install -r requirements.txt
```

The notebook uses Foundry SDK 2.x `AIProjectClient` with
`DefaultAzureCredential`. Files, fine-tuning jobs, deployment discovery, and
inference all use one Foundry project endpoint. It does not use API keys,
construct an OpenAI client directly, call manual management routes, or require a
parent Azure AI Services account endpoint.

The notebook can also use privacy-preserving edge frames. This removes the
original photographic person/face content and timestamp overlay while retaining
ordered pose, object, and motion cues for action recognition.

__Required Services:__
- A Microsoft Foundry project in a region that supports global vision
  fine-tuning, with:
  - A GPT-4.1 base deployment
  - A deployment of the completed fine-tuned model
- Azure CLI authentication available to `DefaultAzureCredential`

__Optional Services:__
- Azure AI Foundry for managing fine-tuning in the UI
- An Azure Storage Account

Rename `.env.template` to `.env` and set `AZURE_AI_PROJECT_ENDPOINT`. Run
`az login` before executing the notebook. No API key is required.

Navigate to the video fine-tuning notebook:

- [Fine-Tuning GPT-4.1 for Action Recognition in Video Clips](fine-tune-aoai-gpt4-1-action-detection.ipynb)

__Note:__ If you encounter the following error: `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`  
In this case, your system is missing the shared `libGL.so.1` library which is required by OpenCV.  
On a Ubuntu system, you can install the missing library as follows:
```bash
sudo apt update
sudo apt install libgl1-mesa-glx
```

## Troubleshooting

### Common Issues

**ImportError: libGL.so.1 not found**
- Install the missing library: `sudo apt install libgl1-mesa-glx` (Ubuntu)
- On other systems, install the OpenGL library for your distribution

**Content Filter Triggered (Images with People)**
- The Fine-Tuning API rejects images containing people by default
- Submit a request to opt out: https://customervoice.microsoft.com/Pages/ResponsePage.aspx
- Select "Modified Content Filtering" for Inferencing and Fine Tuning

**Authentication Error**
- Verify `.env` contains the Foundry project endpoint
- Run `az login` to refresh Azure credentials
- Ensure your identity can access the Foundry project and its deployments

**Training Job Fails**
- Verify video frames are properly extracted and encoded
- Check that JSONL format matches the expected schema
- Ensure images don't exceed size limits

**Quota Exceeded**
- Request additional quota in Azure Portal → Azure OpenAI → Quotas
- Try a different Azure region with available capacity