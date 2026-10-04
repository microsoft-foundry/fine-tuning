import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class StreamingNotebookClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg.get("msg_type") == "stream":
            print(msg.get("content", {}).get("text", ""), end="", flush=True)
        return super().process_message(msg, cell, cell_index)


notebook_names = (
    tuple(sys.argv[1:])
    if len(sys.argv) > 1
    else ("demo.ipynb", "demo_with_python_grader.ipynb")
)

for notebook_name in notebook_names:
    path = ROOT / notebook_name
    print(f"Executing {notebook_name}", flush=True)
    notebook = nbformat.read(path, as_version=4)
    client = StreamingNotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        record_timing=True,
    )
    try:
        client.execute(cwd=str(ROOT))
    finally:
        nbformat.write(notebook, path)
    print(f"Completed {notebook_name}", flush=True)
