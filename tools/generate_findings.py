import html
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(
    r"C:\Users\prakharg\.copilot\session-state\f1833604-4ea0-4404-893e-2421eb3ad912\files\demo-reports"
)
OUTPUT = ROOT / "findings.html"

DEMOS = [
    (
        "agentic-rft.json",
        "Agentic RFT Private Preview",
        "RFT with tool calling, multi-graders, and remote endpoint graders for agentic workflows.",
    ),
    (
        "distilling-sarcasm.json",
        "Distilling Sarcasm",
        "Distill sarcastic response behavior from a larger teacher into a smaller student model.",
    ),
    (
        "dpo-intel-orca.json",
        "DPO Intel Orca",
        "Use preference pairs to improve response quality with Direct Preference Optimization.",
    ),
    (
        "evaluation.json",
        "Multimodal Evaluation",
        "Run audio-emotion and image-caption evaluations using Foundry evaluation jobs.",
    ),
    (
        "image-breed.json",
        "Image Breed Classification",
        "Fine-tune a vision model for dog-breed classification and compare accuracy and latency.",
    ),
    (
        "image-chart.json",
        "Image Chart Analysis",
        "Fine-tune a vision model to improve visual and logical reasoning over charts.",
    ),
    (
        "nl-to-python.json",
        "NL-to-Python Distillation",
        "Distill text-to-Python generation into a smaller model and compare generated code.",
    ),
    (
        "rft-countdown.json",
        "RFT Countdown",
        "Teach mathematical countdown solving with score-model and Python graders.",
    ),
    (
        "rft-math.json",
        "RFT Math Reasoning",
        "Improve advanced mathematical reasoning with reinforcement fine-tuning.",
    ),
    (
        "sft-bug-detection.json",
        "SFT Bug Detection",
        "Train a smaller model to identify, explain, and repair subtle code bugs.",
    ),
    (
        "sft-cnn.json",
        "SFT CNN/DailyMail",
        "Fine-tune a model for concise, factual news summarization.",
    ),
    (
        "sft-pubmed.json",
        "SFT PubMed Summarization",
        "Fine-tune a model to summarize medical research papers.",
    ),
    (
        "synthetic-tooluse.json",
        "Synthetic Data Tool Use",
        "Generate synthetic tool-call traces, fine-tune a student, and evaluate structural correctness.",
    ),
    (
        "traces-distillation.json",
        "Traces Distillation",
        "Distill a hosted agent's observed production tool-use traces into a smaller model.",
    ),
    (
        "video-ft.json",
        "Video Action Recognition",
        "Fine-tune a vision model on extracted video frames for human-action classification.",
    ),
    (
        "zava-retail.json",
        "Zava Retail Agent",
        "Train and validate a policy-compliant retail agent using SFT, RFT, and remote tools.",
    ),
]

RESULTS = {
    "agentic-rft.json": "Partial: endpoint-grader tool-calling RFT succeeded; countdown parsing and GPT-5 multi-grader service failures remained terminal.",
    "distilling-sarcasm.json": "Succeeded: held-out score improved 3.1 to 5.2 and pass rate improved 40% to 80%.",
    "dpo-intel-orca.json": "Succeeded after moving from Sweden Central to East US 2; groundedness improved 3.33 to 4.00.",
    "evaluation.json": "Completed: image evaluation passed 10/10; audio executed successfully but passed 1/10 quality criteria.",
    "image-breed.json": "Succeeded: breed accuracy remained 93.75% while mean latency improved from 2147.9 ms to 1563.6 ms, a 27.2% reduction.",
    "image-chart.json": "Succeeded: ChartQA accuracy improved from 88% to 90%.",
    "nl-to-python.json": "Succeeded after moving from Sweden Central to East US 2; deployed inference passed and the fine-tuned output was preferred for conciseness.",
    "rft-countdown.json": "Succeeded: both grader variants solved 3/3 cases versus 1/3 for the baseline.",
    "rft-math.json": "Partial: training and validation succeeded with reward improving 0.05 to 0.30; deployment was blocked by Fireworks quota and regional model scoping.",
    "sft-bug-detection.json": "Succeeded: score improved 8.43 to 8.80 and pass rate improved 80% to 90%.",
    "sft-cnn.json": "Succeeded: fine-tuned news summarization deployment passed factual inference validation.",
    "sft-pubmed.json": "Succeeded: deployed medical summarization retained the study design, biomarkers, brain regions, accuracy, and conclusion.",
    "synthetic-tooluse.json": "Succeeded: structural tool-use score improved 7.0 to 9.0 and pass rate improved 66.7% to 88.9%.",
    "traces-distillation.json": "Succeeded after creating a hosted trace source: structural tool-call score improved from 7.50 to 9.50 and pass rate improved from 50% to 90%.",
    "video-ft.json": "Succeeded: action-recognition accuracy improved from 73.3% to 86.7%, with a documented small accepted training subset.",
    "zava-retail.json": "Succeeded with platform limitations: SFT accuracy improved 58.25% to 95.15%; optimized RFT and policy validation completed.",
}


def esc(value):
    return html.escape(str(value), quote=True)


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def first(data, *keys):
    for node in walk(data):
        for key in keys:
            value = node.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def unique(items):
    seen = set()
    result = []
    for item in items:
        marker = json.dumps(item, sort_keys=True, default=str) if not isinstance(item, str) else item
        if marker not in seen:
            seen.add(marker)
            result.append(item)
    return result


def extract_endpoints(data):
    return unique(
        value.rstrip("/")
        for value in strings(data)
        if "services.ai.azure.com/api/projects/" in value
    )


def extract_models(data):
    values = []
    model_keys = {
        "model",
        "model_name",
        "base_model",
        "student_model",
        "teacher_model",
        "fine_tuned_model",
        "fineTunedModel",
    }
    for node in walk(data):
        for key, value in node.items():
            if key in model_keys and isinstance(value, str) and value:
                values.append(value)
    return unique(values)


def extract_jobs(data):
    found = {}
    for node in walk(data):
        job_id = None
        for key in ("id", "job_id", "jobId"):
            value = node.get(key)
            if isinstance(value, str) and value.startswith("ftjob-"):
                job_id = value
                break
        if not job_id:
            continue
        current = found.setdefault(job_id, {"id": job_id})
        for target, keys in {
            "status": ("terminal_status", "status", "state"),
            "model": ("model", "base_model", "model_name"),
            "fine_tuned_model": ("fine_tuned_model", "fineTunedModel"),
            "error": ("error", "failure", "failure_reason"),
            "url": ("portal_url", "job_url", "url", "link"),
        }.items():
            if not current.get(target):
                for key in keys:
                    if node.get(key) not in (None, "", {}, []):
                        current[target] = node[key]
                        break
    endpoints = extract_endpoints(data)
    endpoint = endpoints[-1] if endpoints else None
    for job in found.values():
        if not job.get("url") and endpoint:
            job["url"] = f"{endpoint}/openai/v1/fine_tuning/jobs/{job['id']}"
        if isinstance(job.get("error"), dict):
            job["error"] = job["error"].get("message") or json.dumps(job["error"], ensure_ascii=False)
    return list(found.values())


def extract_list(data, keys):
    result = []
    for node in walk(data):
        for key in keys:
            value = node.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        result.append(item)
                    elif isinstance(item, dict):
                        text = item.get("detail") or item.get("message") or item.get("fix") or item.get("result")
                        if text:
                            result.append(str(text))
    return unique(result)


def outcome_class(filename, data):
    status = str(first(data, "status", "outcome") or "").lower()
    if filename in ("agentic-rft.json", "rft-math.json"):
        return "partial", "Partial"
    if "success" in status or "complete" in status:
        return "success", "Succeeded"
    if "block" in status or "fail" in status:
        return "blocked", "Blocked"
    return "partial", "Completed with findings"


def job_rows(jobs):
    if not jobs:
        return '<tr><td colspan="5" class="muted">No fine-tuning job was expected or created.</td></tr>'
    rows = []
    for job in jobs:
        job_id = esc(job["id"])
        link = job_id
        if job.get("url"):
            link = f'<a href="{esc(job["url"])}">{job_id}</a>'
        error = job.get("error") or ""
        rows.append(
            "<tr>"
            f"<td>{link}</td>"
            f"<td>{esc(job.get('model') or 'Not recorded')}</td>"
            f"<td>{esc(job.get('status') or 'Not recorded')}</td>"
            f"<td>{esc(job.get('fine_tuned_model') or '')}</td>"
            f"<td>{esc(error)}</td>"
            "</tr>"
        )
    return "".join(rows)


def list_html(items, empty):
    if not items:
        return f'<p class="muted">{esc(empty)}</p>'
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"


def render_demo(filename, title, purpose, data):
    css_class, label = outcome_class(filename, data)
    endpoints = extract_endpoints(data)
    models = extract_models(data)
    jobs = extract_jobs(data)
    fixes = extract_list(data, ("changes", "fixes", "corrective_changes", "errors_and_fixes", "retries_and_fixes"))
    findings = extract_list(data, ("findings", "warnings", "limitations", "externalPrerequisites"))
    result = RESULTS.get(filename) or first(data, "summary") or "See the execution details below."
    project = "<br>".join(f'<a href="{esc(endpoint)}">{esc(endpoint)}</a>' for endpoint in endpoints)
    if not project:
        project = esc(first(data, "project_name", "resource_name", "account_name") or "Recorded in execution details")
    raw = esc(json.dumps(data, indent=2, ensure_ascii=False))
    return f"""
    <section class="demo {css_class}">
      <div class="demo-head">
        <div><h2>{esc(title)}</h2><p>{esc(purpose)}</p></div>
        <span class="badge">{esc(label)}</span>
      </div>
      <div class="result"><strong>Outcome:</strong> {esc(result)}</div>
      <div class="facts">
        <div><strong>Foundry resource/project</strong><br>{project}</div>
        <div><strong>Models</strong><br>{esc(", ".join(models) if models else "No training model recorded")}</div>
        <div><strong>Fine-tuning jobs</strong><br>{len(jobs)}</div>
      </div>
      <h3>Fine-tuning job history</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>Job</th><th>Base model</th><th>Status</th><th>Fine-tuned model</th><th>Error</th></tr></thead>
        <tbody>{job_rows(jobs)}</tbody>
      </table></div>
      <div class="columns">
        <div><h3>Corrections and fixes</h3>{list_html(fixes, "No corrective source change was required.")}</div>
        <div><h3>Findings and limitations</h3>{list_html(findings, "No additional limitation was recorded.")}</div>
      </div>
      <details><summary>Complete machine-readable execution record</summary><pre>{raw}</pre></details>
    </section>
    """


def main():
    missing = [filename for filename, _, _ in DEMOS if not (REPORT_DIR / filename).exists()]
    if missing:
        raise SystemExit(f"Missing demo reports: {', '.join(missing)}")
    reports = [(item, json.loads((REPORT_DIR / item[0]).read_text(encoding="utf-8"))) for item in DEMOS]
    statuses = [outcome_class(item[0], data)[0] for item, data in reports]
    job_count = sum(len(extract_jobs(data)) for _, data in reports)
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    sections = "".join(render_demo(*item, data) for item, data in reports)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fine-Tuning Demo Execution Findings</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#5e6b82; --line:#d9e1ee; --bg:#f5f7fb; --card:#fff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 "Segoe UI",Arial,sans-serif; }}
    main {{ max-width:1500px; margin:auto; padding:32px; }}
    h1 {{ margin:0 0 8px; font-size:32px; }} h2 {{ margin:0; font-size:22px; }} h3 {{ margin:22px 0 8px; font-size:15px; }}
    p {{ margin:5px 0; }} a {{ color:#075ec7; overflow-wrap:anywhere; }}
    .summary {{ display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:12px; margin:24px 0; }}
    .metric,.demo {{ background:var(--card); border:1px solid var(--line); border-radius:12px; box-shadow:0 3px 12px #1720330a; }}
    .metric {{ padding:18px; }} .metric b {{ display:block; font-size:28px; }} .metric span,.muted {{ color:var(--muted); }}
    .demo {{ margin:18px 0; padding:22px; border-left-width:7px; }} .demo.success {{ border-left-color:#16834b; }} .demo.partial {{ border-left-color:#d18b00; }} .demo.blocked {{ border-left-color:#bd2c34; }}
    .demo-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:24px; }}
    .badge {{ border-radius:999px; padding:5px 11px; font-weight:700; background:#edf2f8; white-space:nowrap; }}
    .success .badge {{ background:#e4f5eb; color:#126b3e; }} .partial .badge {{ background:#fff2cf; color:#815500; }} .blocked .badge {{ background:#fde8e8; color:#9b1c23; }}
    .result {{ margin:18px 0; padding:12px 14px; background:#f7f9fc; border-radius:8px; }}
    .facts {{ display:grid; grid-template-columns:2fr 2fr 1fr; gap:12px; }}
    .facts>div {{ padding:11px; border:1px solid var(--line); border-radius:8px; overflow-wrap:anywhere; }}
    .table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; min-width:900px; }} th,td {{ padding:9px; border:1px solid var(--line); text-align:left; vertical-align:top; }} th {{ background:#edf2f8; }}
    .columns {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }} ul {{ margin-top:6px; padding-left:20px; }}
    details {{ margin-top:20px; }} summary {{ cursor:pointer; font-weight:600; }} pre {{ max-height:600px; overflow:auto; padding:14px; background:#111827; color:#e5e7eb; border-radius:8px; white-space:pre-wrap; overflow-wrap:anywhere; }}
    footer {{ color:var(--muted); margin-top:25px; }}
    @media(max-width:800px) {{ main {{ padding:16px; }} .summary,.facts,.columns {{ grid-template-columns:1fr; }} .demo-head {{ display:block; }} .badge {{ display:inline-block; margin-top:10px; }} }}
  </style>
</head>
<body><main>
  <h1>Fine-Tuning Demo Execution Findings</h1>
  <p>Generated {esc(generated)} from terminal notebook, fine-tuning, deployment, inference, and evaluation records.</p>
  <div class="summary">
    <div class="metric"><b>{len(DEMOS)}</b><span>catalog demos</span></div>
    <div class="metric"><b>{statuses.count("success")}</b><span>successful/completed</span></div>
    <div class="metric"><b>{statuses.count("partial")}</b><span>partial outcomes</span></div>
    <div class="metric"><b>{job_count}</b><span>fine-tuning attempts tracked</span></div>
  </div>
  {sections}
  <footer>Source records: {esc(REPORT_DIR)}. Job links use the authenticated Foundry portal or project API recorded by each execution.</footer>
</main></body></html>
"""
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
