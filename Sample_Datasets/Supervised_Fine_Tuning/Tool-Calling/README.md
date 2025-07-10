# Notice

This is sample training & validation data, not intended for production use.

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




