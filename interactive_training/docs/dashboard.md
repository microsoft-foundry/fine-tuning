# Run Dashboard

A small built-in dashboard visualizes a run's metrics and adapts to the recipe type: for RL runs it shows reward & accuracy curves, KL, group composition, and token counts; for SFT runs it shows train/eval NLL and token counts. Both views also include optimizer stats, step timing, config, and a tail of `logs.log`. The recipe type is auto-detected from each run's `metrics.jsonl`. Run `dashboard_server.py` from the **cookbook directory** (`fine-tuning/interactive_training`), not the parent repository. It scans the default logs root unless overridden ([storage](./storage.md)).

## Start it

From the **cookbook directory**:

```bash
python dashboard_server.py
# → http://127.0.0.1:8000/
```

For the quickstart's explicit `./runs/first-math-check` directory, use `python dashboard_server.py --root ./runs`; the default logs-root scan will not discover that custom location.

> [!WARNING]
> The dashboard is unauthenticated and can serve prompts, responses, identifiers, and dataset text. Keep it on loopback with private forwarding. Stopping it with Ctrl+C does **not** unload Azure sessions. Review [security](#security) before sharing a URL or artifacts.

Useful flags:

```bash
python dashboard_server.py --root ~/interactive-post-training-runs           # logs root to scan (default per platform)
python dashboard_server.py --port 9000                  # change port
python dashboard_server.py --root ~/run-a --root ~/run-b  # scan multiple roots
# Or via env var:
INTERACTIVE_POST_TRAINING_LOGS_ROOT=/some/other/path python dashboard_server.py
```

> [!TIP]
> **VS Code Web / Codespaces / Remote-SSH.** Leave the server on `127.0.0.1` and forward port `8000` through the **Ports** panel. Keep port visibility **Private** and verify who can access the forwarded URL. Pure browser-only `vscode.dev` without an attached compute backend cannot host the dashboard.

## Use it

1. Open `http://127.0.0.1:8000/` in your browser.
2. Pick a run from the **Run** dropdown in the header. Runs are auto-discovered under the logs root (any directory containing `metrics.jsonl` or `config.json`) and sorted most-recent first. The selected run is encoded in the URL hash, so you can bookmark or share a direct link.
3. Charts and the log tail auto-refresh on the cadence chosen in the **Refresh** dropdown (default 15s); use **Reload now** to force a refresh.
4. The **Per-iteration artifacts** card lists `train_iteration_*.html` and `eval_test_iteration_*.html` files for the selected run; click to open.

## Run Overview card

The Run Overview card pulls from [`run_meta.json`](./storage.md#run_metajson) and shows model name, LoRA rank, endpoint host, Azure session id, and — when the run was launched via `load_checkpoint_path` — a **Resumed from** row linking to the source `<session_id>/<checkpoint_name>`. See [continual-fine-tuning.md](./continual-fine-tuning.md).

## Security

The server reads the configured logs roots. For each run, it serves `metrics.jsonl`, `config.json`, `logs.log`, `run_meta.json`, and `.html` files whose names start with `train_iteration_`, `eval_test_iteration_`, or `eval_`. It also serves the bundled dashboard page and discovery metadata (including logs-root paths). Path traversal is rejected; this allowlist is not authentication or redaction.

The dashboard has **no built-in authentication**. Do not expose it to an untrusted
network with `--host 0.0.0.0`; use a private tunnel or authenticated proxy instead.
Run metadata, logs, and HTML artifacts can contain project/session identifiers,
prompts, completions, and dataset text. Use an access-controlled logs directory
and review artifacts before sharing. The server does not redact their contents.
