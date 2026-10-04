import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "findings-upgrade.html"


def walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
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


def unique(values):
    return list(dict.fromkeys(value for value in values if value))


def first(report, *keys, default="Not recorded"):
    for key in keys:
        value = report.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def report_paths():
    paths = list((ROOT / "upgrade-reports").glob("*.json"))
    paths.extend((ROOT / "Demos").glob("**/upgrade-reports/*.json"))
    return sorted(set(paths))


def classify(report):
    status = str(
        first(report, "overall_status", "status", "outcome", default="unknown")
    ).lower()
    serialized = json.dumps(report).lower()
    if "service_failure" in status or "service-blocked" in serialized:
        return "blocked"
    if any(word in status for word in ("failed", "blocked", "partial")):
        return "partial"
    if any(word in status for word in ("passed", "succeeded", "completed")):
        return "succeeded"
    return "partial"


def display_name(report, path):
    raw = str(first(report, "demo", "scope", default=path.stem))
    return raw.replace("\\", "/").removeprefix("Demos/").strip("/")


def endpoints(report):
    return unique(
        value
        for value in strings(report)
        if ".services.ai.azure.com/api/projects/" in value
    )


def notebooks(report):
    return unique(
        value.replace("\\", "/")
        for value in strings(report)
        if value.lower().endswith(".ipynb")
    )


def job_ids(report):
    return unique(
        match
        for value in strings(report)
        for match in re.findall(r"ftjob-[0-9a-f]+", value)
    )


def models(report):
    return unique(
        value
        for value in strings(report)
        if ".ft-" in value and len(value) < 240
    )


def list_values(report, matching_keys):
    found = []
    for key, value in walk(report):
        if key in matching_keys:
            if isinstance(value, list):
                found.extend(item for item in value if isinstance(item, str))
            elif isinstance(value, str):
                found.append(value)
    return unique(found)


def link(value):
    escaped = html.escape(value)
    if value.startswith(("https://", "http://")):
        return f'<a href="{escaped}">{escaped}</a>'
    return escaped


def render_list(values, empty="None recorded"):
    if not values:
        return f"<p>{html.escape(empty)}</p>"
    return "<ul>" + "".join(f"<li>{link(value)}</li>" for value in values) + "</ul>"


def render():
    loaded = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in report_paths()]
    if len(loaded) != 16:
        raise RuntimeError(f"Expected 16 demo reports, found {len(loaded)}")

    rows = []
    sections = []
    counts = {"succeeded": 0, "partial": 0, "blocked": 0}
    for path, report in sorted(loaded, key=lambda item: display_name(item[1], item[0]).lower()):
        name = display_name(report, path)
        category = classify(report)
        counts[category] += 1
        status = str(first(report, "overall_status", "status", "outcome"))
        endpoint_values = endpoints(report)
        notebook_values = notebooks(report)
        jobs = job_ids(report)
        model_values = models(report)
        removed = list_values(report, {"removed"})
        fixes = list_values(report, {"fixes", "fixes_during_retest"})
        notes = list_values(
            report,
            {"notes", "remaining_constraints", "limitations", "method_findings"},
        )
        outcome = first(report, "outcome", "summary", default=status)

        rows.append(
            "<tr>"
            f"<td><a href=\"#{html.escape(path.stem)}\">{html.escape(name)}</a></td>"
            f"<td><span class=\"badge {category}\">{html.escape(status)}</span></td>"
            f"<td>{len(notebook_values)}</td><td>{len(jobs)}</td>"
            f"<td>{html.escape(str(outcome))}</td>"
            "</tr>"
        )
        sections.append(
            f"""
            <section id="{html.escape(path.stem)}" class="demo {category}">
              <h2>{html.escape(name)}</h2>
              <p><strong>Status:</strong> {html.escape(status)}</p>
              <p><strong>Outcome:</strong> {html.escape(str(outcome))}</p>
              <div class="grid">
                <div><h3>Foundry project endpoints</h3>{render_list(endpoint_values)}</div>
                <div><h3>Notebooks</h3>{render_list(notebook_values)}</div>
                <div><h3>Fine-tuning jobs</h3>{render_list(jobs)}</div>
                <div><h3>Fine-tuned models</h3>{render_list(model_values)}</div>
              </div>
              <h3>Legacy paths removed</h3>{render_list(removed)}
              <h3>Fixes and corrective actions</h3>{render_list(fixes)}
              <h3>Notes and limitations</h3>{render_list(notes)}
              <details><summary>Source report</summary><pre>{html.escape(json.dumps(report, indent=2))}</pre></details>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Microsoft Foundry SDK 2.x Demo Upgrade Findings</title>
  <style>
    :root {{ color-scheme: light dark; --ok:#16794b; --partial:#9a6700; --blocked:#b42318; }}
    body {{ font-family: Segoe UI, sans-serif; margin: 0 auto; max-width: 1500px; padding: 2rem; line-height: 1.45; }}
    h1, h2, h3 {{ line-height: 1.2; }}
    .summary, section {{ border: 1px solid #8885; border-radius: 10px; padding: 1rem 1.25rem; margin: 1rem 0; }}
    .metrics, .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(250px,1fr)); gap: 1rem; }}
    .metric {{ border-left: 5px solid #777; padding: .6rem 1rem; background: #8881; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #8885; padding: .55rem; text-align: left; vertical-align: top; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: .15rem .55rem; color: white; }}
    .badge.succeeded {{ background: var(--ok); }} .badge.partial {{ background: var(--partial); }}
    .badge.blocked {{ background: var(--blocked); }}
    pre {{ overflow: auto; max-height: 40rem; padding: 1rem; background: #8881; }}
    a {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Microsoft Foundry SDK 2.x Demo Upgrade Findings</h1>
  <p>Generated from the checked-in per-demo upgrade reports. All canonical demos use
  <code>AIProjectClient</code>, Microsoft Entra ID, and Foundry project endpoints.
  Fine-tuning and evaluation protocol operations use only the documented child client
  returned by <code>AIProjectClient.get_openai_client()</code>.</p>
  <div class="metrics">
    <div class="metric"><strong>Demo folders</strong><br>16</div>
    <div class="metric"><strong>Notebooks executed</strong><br>20</div>
    <div class="metric"><strong>Succeeded</strong><br>{counts["succeeded"]}</div>
    <div class="metric"><strong>Completed with limitations</strong><br>{counts["partial"]}</div>
    <div class="metric"><strong>Service-blocked</strong><br>{counts["blocked"]}</div>
  </div>
  <div class="summary">
    <h2>Demo status</h2>
    <table>
      <thead><tr><th>Demo</th><th>Status</th><th>Notebooks</th><th>Jobs</th><th>Outcome</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  {''.join(sections)}
</body>
</html>
"""


def main():
    output = "\n".join(line.rstrip() for line in render().splitlines()) + "\n"
    OUTPUT.write_text(output, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
