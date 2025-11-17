# Zava Retail Agent

A retail customer service agent built using Azure OpenAI, designed to handle customer inquiries about orders, returns, and product information.

## Overview

This project demonstrates a fine-tuned AI agent for retail operations that can:
- Query order information
- Process return requests
- Answer product-related questions
- Follow company policies

## Project Structure

```
ZavaRetailAgent/
├── demo.ipynb              # Jupyter notebook with demo and examples
├── requirements.txt        # Python dependencies
├── data/                   # Training and test data
│   ├── db.json            # Sample retail database
│   ├── openapi_with_policy.json  # API specifications
│   ├── policy.md          # Company policy documentation
│   ├── sft_train.jsonl    # Supervised fine-tuning training data
│   ├── sft_test.jsonl     # Supervised fine-tuning test data
│   ├── rft_train.jsonl    # Reinforcement fine-tuning training data
│   └── rft_test.jsonl     # Reinforcement fine-tuning test data
└── tools/                  # Agent tools and utilities
    └── retail_agent.py    # Core retail agent implementation
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Demos/ZavaRetailAgent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` with your actual values:
   - `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
   - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
   - `AZURE_OPENAI_DEPLOYMENT_NAME`: Your deployment name
   - `AZURE_OPENAI_API_VERSION`: API version (default: 2024-08-01-preview)

4. **Run the demo**
   ```bash
   jupyter notebook demo.ipynb
   ```

## Features

- **Order Management**: Query order status, tracking, and details
- **Returns Processing**: Handle return requests following company policy
- **Product Information**: Provide product details and availability
- **Policy Compliance**: Ensures responses align with company guidelines

## Training Data

The project includes both supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) datasets:
- Training and test sets for model development
- Configuration files for grading and tools
- Sample conversations for testing

## Requirements

- Python 3.12+
- Azure OpenAI Service access
- Required Python packages (see `requirements.txt`)

## License

See the main repository for license information.
