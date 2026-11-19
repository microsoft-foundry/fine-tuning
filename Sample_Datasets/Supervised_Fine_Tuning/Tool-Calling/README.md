# Notice

This is sample training & validation data, not intended for production use.

## Upstream source and retained license

The published reference is the [Microsoft fine-tuning Tool-Calling directory](https://github.com/microsoft-foundry/fine-tuning/tree/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/Sample_Datasets/Supervised_Fine_Tuning/Tool-Calling)
at Git revision `cc042d12f6704be36c70f9f4b2457aeaa12b97c7`.
The [upstream repository license](https://github.com/microsoft-foundry/fine-tuning/blob/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/LICENSE)
is MIT, with `Copyright (c) 2025 Azure AI Foundry`. Its complete notice is
retained in [LICENSE.md](LICENSE.md); keep it with redistributed copies.

| Published source file | Git blob SHA-1 | Source bytes |
|---|---|---:|
| [stock_toolcall_training.jsonl](https://github.com/microsoft-foundry/fine-tuning/blob/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/Sample_Datasets/Supervised_Fine_Tuning/Tool-Calling/stock_toolcall_training.jsonl) | `b864adcdd3decff9e4332f572f51fd166c093ad1` | 96351 |
| [stock_toolcall_validation.jsonl](https://github.com/microsoft-foundry/fine-tuning/blob/cc042d12f6704be36c70f9f4b2457aeaa12b97c7/Sample_Datasets/Supervised_Fine_Tuning/Tool-Calling/stock_toolcall_validation.jsonl) | `a22acedf0fc5e42d1a6bbf1a03eeb5c0bbe67691` | 11222 |

These are upstream reference identifiers, not newly computed local hashes or
an assertion that checkout bytes have been compared. This notice update does
not change sample records or establish provenance for other tool-use datasets.

# Dataset Summary

## Stock Market Query Dataset

This dataset contains  fine tuning training and validation dataset, for stock market-related queries using two primary tools:

1. **get_current_stock_price** - Retrieves the current trading price for a given stock symbol
2. **get_last_nday_stock_price** - Retrieves historical stock price data for specified time periods (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

This dataset is intended to demonstrate how to use tool calls in a fine tuning training data. The tools called do not integrate with real external APIs.

## Key Characteristics

### System Behavior
The assistant follows a strict protocol of not making assumptions about stock ticker symbols. When the exact symbol isn't known, it asks for clarification rather than guessing.

### Query Types Covered
- Current stock prices (Goldman Sachs, Home Depot, Intel)
- Historical price data (highest/lowest prices for specific periods)
- Price ranges for trading days/weeks
- Requests for companies with unclear or fictional names

### Response Patterns
This dataset fine tunes the model to know when to make tool calls (and which tools to call), and when to fail to make calls (because the company does not exist).




