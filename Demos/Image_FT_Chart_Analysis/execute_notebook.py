import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class StreamingNotebookClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg.get("msg_type") == "stream":
            text = msg.get("content", {}).get("text", "")
            if text:
                print(text, end="", flush=True)
        return super().process_message(msg, cell, cell_index)


notebook_path = Path("fine-tune-aoai-gpt4-1-for-chart-analysis.ipynb")
notebook = nbformat.read(notebook_path, as_version=4)
client = StreamingNotebookClient(
    notebook,
    timeout=None,
    kernel_name="python3",
    resources={"metadata": {"path": str(notebook_path.parent.resolve())}},
    record_timing=True,
)

try:
    print(f"Executing {notebook_path} with {sys.executable}", flush=True)
    client.execute()
finally:
    nbformat.write(notebook, notebook_path)
