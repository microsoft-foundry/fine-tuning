# Fine-Tuning Demos

This directory contains cookbooks demonstrating various fine-tuning techniques on Microsoft Foundry.

## Overview

### [Agentic_RFT_PrivatePreview](Agentic_RFT_PrivatePreview/)
**Techniques**: Reinforcement Fine-Tuning (RFT) with Tool Calling & Endpoint Graders  
**Use Case**: Training reasoning models for agentic scenarios with custom tool use  
**Dataset**: Custom tool calling and grading examples  
**Products/SDKs**: Microsoft Foundry, Azure OpenAI API, GPT-5/o4-mini  
**What it shows**: Configure endpoint graders, define tools for chain-of-thought reasoning, train models for agentic workflows

### [DistillingSarcasm](DistillingSarcasm/)
**Technique**: Distillation (SFT)  
**Use Case**: Teaching smaller models to replicate larger model behaviors (sarcasm generation)  
**Dataset**: Human-curated sarcastic Q&A examples  
**Products/SDKs**: Microsoft Foundry, Azure OpenAI API (o3, o4-mini, gpt-4.1, gpt-4o series)  
**What it shows**: Build graders, benchmark models, select teacher/student, distill knowledge, and evaluate improvement

### [DPO_Intel_Orca](DPO_Intel_Orca/)
**Technique**: Direct Preference Optimization (DPO)  
**Use Case**: Training models to prefer high-quality responses over lower-quality alternatives  
**Dataset**: Intel Orca DPO Pairs (preference pairs covering math, reasoning, comprehension)  
**Products/SDKs**: Microsoft Foundry, Azure AI Projects SDK, Azure AI Evaluation SDK  
**What it shows**: Upload datasets, create DPO fine-tuning job, monitor training, deploy model, inference, and evaluate improvements

### [Evaluation](Evaluation/)
**Technique**: Model Evaluation  
**Use Case**: Evaluating multimodal models on audio and image tasks  
**Dataset**: CREMA-D (audio emotion), Conceptual Captions (images)  
**Products/SDKs**: Azure OpenAI, Azure AI Evaluation SDK  
**What it shows**: Upload evaluation datasets, configure score model graders, run multimodal evaluations

### [Image_Breed_Classification_FT](Image_Breed_Classification_FT/)
**Technique**: Vision Fine-Tuning (LoRA SFT)  
**Use Case**: Multi-class image classification (120 dog breeds)  
**Dataset**: Stanford Dogs (50 images per breed, 6,000 total)  
**Products/SDKs**: Azure OpenAI (gpt-4o), Microsoft Foundry  
**What it shows**: Compare zero-shot vs fine-tuned VLM vs CNN baseline, measure accuracy and latency improvements

### [Image_FT_Chart_Analysis](Image_FT_Chart_Analysis/)
**Technique**: Vision Fine-Tuning  
**Use Case**: Chart analysis with visual and logical reasoning  
**Dataset**: ChartQA (chart images with Q&A pairs)  
**Products/SDKs**: Azure OpenAI (GPT-4.1), Microsoft Foundry  
**What it shows**: Prepare chart data, fine-tune vision model, evaluate chart comprehension improvements

### [RFT_Countdown](RFT_Countdown/)
**Technique**: Reinforcement Fine-Tuning (RFT)  
**Use Case**: Teaching models to solve countdown math puzzles  
**Dataset**: Custom countdown puzzle data  
**Products/SDKs**: Microsoft Foundry, Azure OpenAI  
**What it shows**: Define Python graders, create RFT jobs, monitor reinforcement learning progress

### [RFT_Math_Reasoning](RFT_Math_Reasoning/)
**Technique**: Reinforcement Fine-Tuning (RFT)  
**Use Case**: Advanced mathematical reasoning and problem-solving  
**Dataset**: OpenR1-Math-220k  
**Products/SDKs**: Microsoft Foundry, Azure AI Projects SDK  
**What it shows**: Upload RFT datasets, create grading function for mathematical reasoning, configure and launch RFT job, monitor training progress, deploy model, and test advanced mathematical problem-solving capabilities

### [SFT_CNN_DailyMail](SFT_CNN_DailyMail/)
**Technique**: Supervised Fine-Tuning (SFT)  
**Use Case**: News article summarization  
**Dataset**: CNN/DailyMail (2,504 article-summary pairs)  
**Products/SDKs**: Microsoft Foundry, Azure AI Projects SDK  
**What it shows**: Upload datasets, create SFT job, monitor training, deploy model, and test news summarization

### [SFT_PubMed_Summarization](SFT_PubMed_Summarization/)
**Technique**: Supervised Fine-Tuning (SFT)  
**Use Case**: Medical research paper summarization  
**Dataset**: PubMed (6,655 article-abstract pairs)  
**Products/SDKs**: Microsoft Foundry, Azure AI Projects SDK  
**What it shows**: Upload datasets, create SFT job, monitor training, deploy model, and test medical summarization

### [Video_FT_Action_Recognition](Video_FT_Action_Recognition/)
**Technique**: Vision Fine-Tuning  
**Use Case**: Human action recognition in video clips  
**Dataset**: UCF101 (101 action categories, 13,320 video clips)  
**Products/SDKs**: Azure OpenAI (GPT-4.1), Microsoft Foundry  
**What it shows**: Process video frames, fine-tune vision model for action detection, evaluate video understanding

### [ZavaRetailAgent](ZavaRetailAgent/)
**Technique**: Supervised Fine-Tuning (SFT) & Reinforcement Fine-Tuning (RFT) - Ignite 2025 Demo  
**Use Case**: Retail customer service agent for orders, returns, and product inquiries  
**Dataset**: Custom retail conversation data with tool calls  
**Products/SDKs**: Azure OpenAI, Microsoft Foundry  
**What it shows**: Build a retail agent with tool use, train with SFT and RFT, ensure policy compliance

---

*Note: Each demo includes a complete notebook, dataset, requirements, and detailed README.*
