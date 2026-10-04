from pathlib import Path

import nbformat
from nbclient import NotebookClient


directory = Path(__file__).resolve().parent
notebook_path = directory / "fine-tune-aoai-gpt4-1-action-detection.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=None,
    kernel_name="python3",
    resources={"metadata": {"path": str(directory)}},
    allow_errors=False,
)

try:
    client.execute(cwd=str(directory))
finally:
    nbformat.write(notebook, notebook_path)
