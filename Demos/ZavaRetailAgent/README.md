# Zava Retail Agent

A retail customer service agent built with Microsoft Foundry SDK 2.x and Microsoft Entra ID.

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
   
   Copy `.env.example` to `.env` and configure the Foundry project:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` with:
   - `FOUNDRY_PROJECT_ENDPOINT`: Project endpoint ending in `/api/projects/<project>`
   - `FOUNDRY_MODEL_NAME`: Model deployment name
   - `MCP_SERVER_URL`: Zava retail MCP endpoint ending in `/mcp`
   - `TOOLS_SERVER_URL`: Remote tool endpoint used by RFT rollouts

   Authenticate with Entra ID:
   ```bash
   az login
   ```

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
- Microsoft Foundry project access with an Entra role assignment
- Required Python packages (see `requirements.txt`)

## License

See the main repository for license information.

## Troubleshooting

### Common Issues

**Authentication Error**
- Verify `FOUNDRY_PROJECT_ENDPOINT` is a project endpoint, not an account endpoint
- Run `az login` to refresh Azure credentials
- Check that `FOUNDRY_MODEL_NAME` matches a project deployment

**Tool Execution Fails**
- Ensure `data/db.json` contains valid sample data
- Check that tool definitions in `openapi_with_policy.json` are correct
- Verify policy.md is accessible

**Training Job Fails**
- Verify JSONL format matches expected schema for SFT/RFT
- Check that tool call format is correct
- Ensure training data follows company policy constraints

**Quota Exceeded**
- Request additional quota for the Foundry resource in Azure Portal
- Try a different Azure region with available capacity
