import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


path = Path("Text_to_Python_Fine_Tuning.ipynb")
notebook = nbformat.read(path, as_version=4)


def remove_transport_logs():
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            output["text"] = "".join(
                line
                for line in output.get("text", "").splitlines(keepends=True)
                if "[INFO]" not in line and "Request URL:" not in line
            )


parser = argparse.ArgumentParser()
parser.add_argument("--sanitize-only", action="store_true")
args = parser.parse_args()

if not args.sanitize_only:
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent.resolve())}},
        allow_errors=False,
    )
    client.execute()

remove_transport_logs()
nbformat.write(notebook, path)
