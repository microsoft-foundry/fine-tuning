# Fine-Tuning Demos

This directory contains cookbooks demonstrating various fine-tuning techniques on Microsoft Foundry.

## Overview

### [DPO_Intel_Orca](DPO_Intel_Orca/)
**Technique**: Direct Preference Optimization (DPO)  
**Use Case**: Training models to prefer high-quality responses over lower-quality alternatives  
**Dataset**: Intel Orca DPO Pairs (preference pairs covering math, reasoning, comprehension)  
**Products/SDKs**: Microsoft Foundry, Azure AI Projects SDK, Azure AI Evaluation SDK  
**What it shows**: Upload datasets, create DPO fine-tuning job, monitor training, deploy model, inference, and evaluate improvements

### [SFT_CNN_DailyMail](SFT_CNN_DailyMail/)
**Technique**: Supervised Fine-Tuning (SFT)  
**Use Case**: News article summarization  
**Dataset**: CNN/DailyMail (article-summary pairs)  
**Products/SDKs**: Microsoft Foundry, Azure AI Projects SDK  
**What it shows**: Upload datasets, create SFT job, monitor training, deploy model, and test news summarization

### [SFT_PubMed_Summarization](SFT_PubMed_Summarization/)
**Technique**: Supervised Fine-Tuning (SFT)  
**Use Case**: Medical research paper summarization  
**Dataset**: PubMed Medical Research (article-abstract pairs)  
**Products/SDKs**: Microsoft Foundry, Azure AI Projects SDK  
**What it shows**: Upload datasets, create SFT job, monitor training, deploy model, and test medical summarization

---

*Note: Each demo includes a complete notebook, dataset, requirements, and detailed README.*
