# 🚀 MSBuild RFT Demo Notebook

Welcome to the Responsible Finetuning (RFT) Demo for Microsoft Build! This guide will walk you through setting up your environment and running the `demo.ipynb` notebook.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
*   Python (3.8 or higher recommended)
*   pip (Python package installer)
*   Jupyter Notebook or JupyterLab

---

## 🛠️ Environment Setup

Follow these steps to prepare your environment:

1.  **Clone the Repository (if applicable)**
    If you haven't already, clone the repository containing this demo to your local machine.
    ```bash
    # Example: git clone <repository-url>
    # cd <repository-folder>/MSBuildRFTDemo
    ```

2.  **Install Dependencies** 📦
    Open your terminal or command prompt in the `MSBuildRFTDemo` folder (where `demo.ipynb` and `requirements.txt` are located) and run:
    ```bash
    pip install -r requirements.txt
    ```
    This will install all the necessary Python libraries for the RFT demo.

3.  **Configure Environment Variables** 🔑
    You'll need to set up API keys and other configurations. Create a file named `.env` in the `MSBuildRFTDemo` folder and add the following content, replacing the placeholders with your actual values:
    ```env
    AZURE_API_KEY="YOUR_AZURE_OPENAI_API_KEY_HERE"
    AZURE_API_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT_HERE"
    API_VERSION="YOUR_AZURE_OPENAI_API_VERSION_HERE" # e.g., 2023-07-01-preview
    ```
    ⚠️ **Important:** Do not commit your `.env` file to version control if it contains sensitive credentials. Ensure it's listed in your `.gitignore` file.

---

## 🌟 Running the Demo Notebook

Once your environment is set up, you can run the demo:

1.  **Launch Jupyter Notebook** 🚀
    In your terminal, navigate to the `MSBuildRFTDemo` folder and start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
    Or, if you prefer JupyterLab:
    ```bash
    jupyter lab
    ```
    This will open Jupyter in your web browser.

2.  **Open `demo.ipynb`** 📂
    In the Jupyter interface, click on `demo.ipynb` to open the notebook.

3.  **Execute the Cells** ▶️
    Run the cells in the notebook sequentially. You can do this by:
    *   Clicking the "Run" button in the toolbar for each cell.
    *   Using the keyboard shortcut `Shift + Enter` to run the current cell and move to the next.
    *   Selecting "Cell" > "Run All" from the menu bar to run all cells at once.

    The notebook will guide you through:
    *   📊 Loading and preparing data for finetuning.
    *   🛡️ Applying Responsible AI techniques during the finetuning process.
    *   🤖 Evaluating the performance and safety of the finetuned model.
    *   📈 Visualizing the results and RFT metrics.

---

## 📝 Notes & Troubleshooting

*   **API Key Errors**: If you encounter errors related to API keys, double-check your `.env` file for typos or incorrect values. Ensure the keys have the necessary permissions for Azure OpenAI and any other services used.
*   **Missing Modules**: If you get `ModuleNotFoundError`, ensure you have run `pip install -r requirements.txt` in the correct Python environment.
*   **Kernel Issues**: If the notebook kernel has problems, try restarting it ("Kernel" > "Restart" in the Jupyter menu).
*   **RFT Specifics**: This demo utilizes specific RFT tools and configurations. Refer to the comments within `demo.ipynb` for detailed explanations of each step.

---

Thank you for checking out the RFT Demo at Microsoft Build! ✨