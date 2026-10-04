from pathlib import Path

import nbformat
from nbclient import NotebookClient


path = Path("Text_to_Python_Fine_Tuning.ipynb")
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=None,
    kernel_name="python3",
    resources={"metadata": {"path": str(path.parent.resolve())}},
    allow_errors=False,
)

try:
    client.execute()
finally:
    nbformat.write(notebook, path)
