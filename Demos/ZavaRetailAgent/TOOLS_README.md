# Analysis Tools Documentation

This document provides an overview of all analysis and utility tools in the `tools/` directory, including how to invoke them and what data they generate.

---

## Table of Contents

1. [Data Analysis Tools](#data-analysis-tools)
2. [RFT Evaluation Tools](#rft-evaluation-tools)
3. [Testing Tools](#testing-tools)
4. [Utility Tools](#utility-tools)

---

## Data Analysis Tools

### 1. `analyze_db_json.py`

**Purpose**: Analyzes the product database JSON file to understand the data structure, statistics, and relationships.

**Usage**:
```bash
python tools/analyze_db_json.py
```

**Output**:
- Console output with database statistics:
  - Total number of users, products, items, orders
  - Product distribution analysis
  - Order status breakdown
  - Item option variations per product
  - Payment method statistics
- Visual understanding of the database schema

**Use Case**: Use this when you need to understand the structure of `data/db.json` or verify data integrity before running experiments.

---

### 2. `analyze_synthetic_datagen.py`

**Purpose**: Analyzes synthetic data generation output to assess quality, diversity, and distribution of generated conversations.

**Usage**:
```bash
python tools/analyze_synthetic_datagen.py
```

**Output**:
- Statistics on generated conversations
- Distribution analysis of conversation types
- Quality metrics for synthetic data

**Use Case**: Validate the quality and diversity of synthetically generated training data before using it for fine-tuning.

---

## RFT Evaluation Tools

### 3. `analyze_rft_eval.py`

**Purpose**: Downloads RFT (Reinforced Fine-Tuning) evaluation data from Azure OpenAI service with all detailed fields.

**Usage**:
```bash
python tools/analyze_rft_eval.py
```

**Configuration**: Requires Azure OpenAI credentials and evaluation run ID in the script.

**Output**:
- **Directory**: `analysis_charts/rft_eval/data/`
- **Files**:
  - `Step_N.json` - Evaluation data for each step (N = 0, 3, 6, 9, 12, 15, 18, etc.)
  - `combined_data.json` - All steps combined
  - `download_summary.json` - Metadata about the download

**Data Fields Extracted**:
- Messages (conversation history)
- Tools (available functions)
- Model output (assistant responses)
- Reference data:
  - `reference_tool_calls`
  - `reference_policy_args`
  - `reference_policy_outcome`
  - `reference_developer_message`
  - `reference_user_message`
- Evaluation scores
- Trace information
- Sample input/output

**Use Case**: First step in RFT evaluation pipeline - download all evaluation data from Azure for offline analysis.

---

### 4. `analyze_rft_results.py`

**Purpose**: Comprehensive offline analysis of downloaded RFT evaluation data, generating visualizations and statistics.

**Usage**:
```bash
python tools/analyze_rft_results.py
```

**Prerequisites**: Must run `analyze_rft_eval.py` first to download data.

**Output**:
- **Directory**: `analysis_charts/rft_eval/`
- **Charts Generated** (7 total):
  1. `score_progression.png` - Line chart showing mean score progression across steps
  2. `pass_rate_progression.png` - Bar chart with gradient colors showing pass rates
  3. `score_distributions.png` - Grid of histograms (one per step) showing score distributions
  4. `outcome_breakdown.png` - Stacked bar chart (perfect/partial/fail breakdown)
  5. `improvement_delta.png` - Percentage improvement from baseline (Step 0)
  6. `scenario_heatmap.png` - Heatmap showing 10 scenarios × N steps
  7. `scenario_trajectories.png` - Line plot showing each scenario's score trajectory

**Scoring System**:
- `0.0` = Got eligibility and policy reason wrong
- `0.5` = Got eligibility correct but no policy reason
- `1.0` = Got both eligibility and policy reason correct

**Features**:
- Dynamic grid layout (automatically adjusts for any number of steps)
- Groups data into 10 unique scenarios
- Tracks performance across all training steps
- Statistical analysis (mean, median, pass rates)

**Use Case**: Get a comprehensive overview of RFT training performance across all scenarios and steps.

---

### 5. `analyze_rft_test_scenario.py`

**Purpose**: Deep-dive analysis of a single scenario across all RFT training steps with detailed visualizations and full output comparison.

**Usage**:
```bash
# By scenario ID (0-9)
python tools/analyze_rft_test_scenario.py --scenario-id 9

# By search text
python tools/analyze_rft_test_scenario.py --scenario "electric kettle"
```

**Prerequisites**: Must run `analyze_rft_eval.py` first to download data.

**Output**:
- **Directory**: `analysis_charts/rft_eval/scenario_analysis/`
- **Files Generated**:
  1. `scenario_N_summary.csv` - Tabular summary of all variations across steps
  2. `scenario_N_detailed.json` - Full JSON data for the scenario
  3. `scenario_N_scores.png` - Line plot + box plot of score progression
  4. `scenario_N_detailed_table.png` - Score progression table + example comparison
  5. `scenario_N_worst_output.txt` - Full text of worst response (Step 0)
  6. `scenario_N_best_output.txt` - Full text of best response (final step)
  7. `scenario_N_comparison.txt` - Side-by-side comparison of worst vs best

**Features**:
- Analyzes 5 variations per scenario per step
- Extracts and formats model outputs properly (handles Unicode, line breaks)
- Shows score legend in visualizations
- Color coding: red for worst, green for best
- Displays full content without truncation in text files
- Console output with detailed per-variation breakdown

**Use Case**: 
- Investigate specific scenario performance in detail
- Compare worst vs best model responses
- Understand how a particular use case improved through training
- Debug issues with specific test cases

**Example Output**:
```
Performance Summary:
   Step 0 avg: 0.300
   Step 18 avg: 1.000
   Improvement: +233.3%
```

---

### 6. `analyze_eval_run.py`

**Purpose**: Analyzes a single evaluation run from Azure OpenAI.

**Usage**:
```bash
python tools/analyze_eval_run.py
```

**Configuration**: Requires Azure OpenAI credentials and run ID in the script.

**Output**:
- Evaluation metrics and statistics
- Performance breakdown

**Use Case**: Quick analysis of a single evaluation run without full RFT pipeline context.

---

## Testing Tools

### 7. `test_retail_agent.py`

**Purpose**: Integration tests for the retail agent functionality.

**Usage**:
```bash
python tools/test_retail_agent.py
```

**Output**:
- Test results (pass/fail)
- Console output with test details

**Use Case**: Verify retail agent functionality before/after changes.

---

### 8. `test_mcp_connectivity.py`

**Purpose**: Tests Model Context Protocol (MCP) connectivity and functionality.

**Usage**:
```bash
python tools/test_mcp_connectivity.py
```

**Output**:
- Connection status
- MCP server response validation

**Use Case**: Diagnose MCP connection issues or verify MCP server setup.

---

## Utility Tools

### 9. `retail_agent.py`

**Purpose**: Core retail agent implementation with tools for order management, returns, exchanges, etc.

**Usage**: 
- Imported by other scripts
- Can be run standalone for testing

**Features**:
- Tool definitions for retail operations
- Policy verification logic
- Order lookup and modification functions

**Use Case**: Core implementation referenced by training and evaluation scripts.

---

### 10. `convert_to_eval.py`

**Purpose**: Converts training data format to evaluation format required by Azure OpenAI evaluation API.

**Usage**:
```bash
python tools/convert_to_eval.py
```

**Output**:
- Converted evaluation dataset files

**Use Case**: Prepare training data for evaluation runs.

---

## Recommended Workflow

### For RFT Evaluation Analysis:

1. **Download Data**:
   ```bash
   python tools/analyze_rft_eval.py
   ```
   - Downloads all evaluation data from Azure
   - Creates `analysis_charts/rft_eval/data/` directory with Step files

2. **Overall Analysis**:
   ```bash
   python tools/analyze_rft_results.py
   ```
   - Generates 7 comprehensive charts
   - Provides high-level performance overview

3. **Scenario Deep Dive**:
   ```bash
   python tools/analyze_rft_test_scenario.py --scenario-id 9
   ```
   - Analyze specific scenarios of interest
   - Compare worst vs best responses
   - Get detailed performance metrics

### For Data Exploration:

1. **Understand Database**:
   ```bash
   python tools/analyze_db_json.py
   ```

2. **Validate Synthetic Data** (if using):
   ```bash
   python tools/analyze_synthetic_datagen.py
   ```

---

## Output Directory Structure

```
analysis_charts/
└── rft_eval/
    ├── data/
    │   ├── Step_0.json
    │   ├── Step_3.json
    │   ├── Step_6.json
    │   ├── Step_9.json
    │   ├── Step_12.json
    │   ├── Step_15.json
    │   ├── Step_18.json
    │   ├── combined_data.json
    │   └── download_summary.json
    ├── scenario_analysis/
    │   ├── scenario_N_summary.csv
    │   ├── scenario_N_detailed.json
    │   ├── scenario_N_scores.png
    │   ├── scenario_N_detailed_table.png
    │   ├── scenario_N_worst_output.txt
    │   ├── scenario_N_best_output.txt
    │   └── scenario_N_comparison.txt
    ├── score_progression.png
    ├── pass_rate_progression.png
    ├── score_distributions.png
    ├── outcome_breakdown.png
    ├── improvement_delta.png
    ├── scenario_heatmap.png
    └── scenario_trajectories.png
```

---

## Dependencies

All tools require the following Python packages:
- `matplotlib` - For visualizations
- `pandas` - For data analysis
- `numpy` - For numerical operations
- `azure-ai-projects` - For Azure OpenAI API access
- `azure-identity` - For Azure authentication
- `json` - For JSON processing
- `pathlib` - For file path handling

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Tips

- **Start with overall analysis** (`analyze_rft_results.py`) to identify interesting scenarios
- **Use scenario analysis** (`analyze_rft_test_scenario.py`) to investigate specific issues
- **Check text files** for full, properly formatted model outputs without truncation
- **Compare worst vs best** to understand quality improvements through training
- **Use scenario heatmap** to identify which scenarios improved most/least
- **Monitor score distributions** to see if the model is becoming more consistent

---

## Troubleshooting

### "No data files found"
- Run `analyze_rft_eval.py` first to download evaluation data

### "Azure authentication failed"
- Verify Azure credentials are set up
- Check subscription and resource access

### "Matplotlib rendering errors"
- Special characters ($, \, _, etc.) in text are now properly escaped
- If issues persist, check the text file outputs instead of PNG charts

### "Step files missing"
- Ensure evaluation runs completed successfully in Azure
- Check that step numbers in `analyze_rft_eval.py` match your training configuration

---

## Contributing

When adding new analysis tools:
1. Follow the naming convention: `analyze_*.py` for analysis tools
2. Add comprehensive docstrings
3. Include usage examples in `--help` output
4. Update this README with tool documentation
5. Generate outputs in the `analysis_charts/` directory
