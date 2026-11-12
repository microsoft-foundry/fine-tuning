"""Exercise only AST-selected notebook functions; never execute paid cells."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/demo.ipynb"


def notebook_cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def cell_source(cell):
    source = cell["source"]
    if isinstance(source, str):
        return source
    return "".join(source) if any(line.endswith(("\n", "\r")) for line in source) else "\n".join(source)


def inference_namespace():
    names = {
        "solve_math", "evaluate_math_deployment", "warn_paid_workflow",
        "validate_math_deployment_identity",
    }
    assignments = {"MODEL", "MODEL_VERSION"}
    parsed = [
        node for cell in notebook_cells() if cell["cell_type"] == "code"
        for node in ast.parse(cell_source(cell)).body
    ]
    functions = [
        node for node in parsed
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    constants = [node for node in parsed if isinstance(node, ast.Assign)
                 and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in assignments]
    assert {node.targets[0].id for node in constants} == assignments
    namespace = {
        "_paid_warning_shown": False,
        "grader_module": SimpleNamespace(grade=Mock(return_value=1.0)),
        "retry": lambda operation, **_kwargs: operation(),
    }
    exec(compile(ast.Module(body=constants + functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def inference_project(deployment):
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=r"\boxed{1}"),
    )])
    create = Mock(return_value=response)
    client = SimpleNamespace(with_options=Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )))
    project = SimpleNamespace(
        deployments=SimpleNamespace(get=Mock(return_value=deployment)),
        get_openai_client=Mock(return_value=client),
    )
    return project, client, create


def test_chat_post_disables_sdk_retries_and_warns_once_without_a_gate(capsys):
    namespace = inference_namespace()
    project, client, create = inference_project(SimpleNamespace(
        name="deployment", model_name=namespace["MODEL"], model_version=namespace["MODEL_VERSION"],
    ))
    row = {"messages": [{"role": "user", "content": "compute"}], "answer": "1"}
    result = namespace["evaluate_math_deployment"](
        project, "deployment", [row, row], role="baseline",
    )
    assert result["correct"] == result["total"] == 2
    assert result["verified_identity"] == {
        "name": "deployment", "model": namespace["MODEL"],
        "version": namespace["MODEL_VERSION"], "role": "baseline",
    }
    project.deployments.get.assert_called_once_with("deployment")
    assert create.call_count == 2
    assert client.with_options.call_count == 2
    for call in client.with_options.call_args_list:
        assert call.kwargs == {"max_retries": 0}
    create.assert_called_with(
        model="deployment", messages=row["messages"], max_completion_tokens=20_000,
    )
    warning = capsys.readouterr().out
    assert warning.count("This workflow uses paid cloud resources and may incur charges.") == 1
    assert "baseline inference" in warning and "endpoint evaluation" in warning


def test_transient_chat_error_is_propagated_after_exactly_one_paid_post():
    class TransientError(Exception):
        status_code = 500

    namespace = inference_namespace()
    create = Mock(side_effect=TransientError("response lost"))
    client = SimpleNamespace(with_options=Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )))
    with pytest.raises(TransientError):
        namespace["solve_math"](
            client,
            SimpleNamespace(name="deployment", model_name=namespace["MODEL"],
                            model_version=namespace["MODEL_VERSION"]),
            {"messages": [], "answer": "1"},
            deployment_name="deployment", role="baseline",
        )
    assert create.call_count == 1
    client.with_options.assert_called_once_with(max_retries=0)
    namespace["grader_module"].grade.assert_not_called()


@pytest.mark.parametrize("role", ["baseline", "fine-tuned"])
@pytest.mark.parametrize("change", [
    "wrong-name", "wrong-model", "wrong-version", "missing-model", "missing-version",
    "blank-model", "blank-version", "padded-version",
])
def test_actual_evaluation_helper_blocks_wrong_or_missing_deployment_identity_before_inference(role, change):
    namespace = inference_namespace()
    job = SimpleNamespace(
        id="offline-job", status="succeeded", model=namespace["MODEL"],
        fine_tuned_model="offline-trained-artifact",
    )
    expected_model = namespace["MODEL"] if role == "baseline" else job.fine_tuned_model
    expected_version = namespace["MODEL_VERSION"] if role == "baseline" else "artifact-version"
    deployment = SimpleNamespace(name="deployment", model_name=expected_model, model_version=expected_version)
    field = "name" if change == "wrong-name" else "model_name" if "model" in change else "model_version"
    if change.startswith("missing"):
        delattr(deployment, field)
    else:
        setattr(deployment, field, "" if change.startswith("blank") else (
            f" {expected_version} " if change == "padded-version" else "unrelated"
        ))
    project, client, create = inference_project(deployment)
    row = {"messages": [], "answer": "1"}
    with pytest.raises(RuntimeError, match="identity is missing or mismatched"):
        namespace["evaluate_math_deployment"](
            project, "deployment", [row], role=role, completed_job=job,
            expected_job_id="offline-job", artifact_version="artifact-version",
        )
    project.get_openai_client.assert_not_called()
    client.with_options.assert_not_called()
    create.assert_not_called()
    namespace["grader_module"].grade.assert_not_called()
    assert namespace["_paid_warning_shown"] is False


@pytest.mark.parametrize("change", [
    "missing-job", "wrong-job-id", "pending-job", "wrong-base", "missing-artifact",
    "missing-expected-job-id", "missing-expected-version",
])
def test_fine_tuned_identity_requires_completed_job_and_independent_expected_version(change):
    namespace = inference_namespace()
    job = SimpleNamespace(
        id="offline-job", status="succeeded", model=namespace["MODEL"],
        fine_tuned_model="offline-trained-artifact",
    )
    deployment = SimpleNamespace(
        name="deployment", model_name=job.fine_tuned_model, model_version="artifact-version",
    )
    expected_job_id, artifact_version = "offline-job", "artifact-version"
    if change == "missing-job":
        job = None
    elif change == "wrong-job-id":
        job.id = "unrelated-job"
    elif change == "pending-job":
        job.status = "running"
    elif change == "wrong-base":
        job.model = "unrelated-model"
    elif change == "missing-artifact":
        job.fine_tuned_model = None
    elif change == "missing-expected-job-id":
        expected_job_id = None
    else:
        artifact_version = None
    project, client, create = inference_project(deployment)
    with pytest.raises(RuntimeError):
        namespace["evaluate_math_deployment"](
            project, "deployment", [{"messages": [], "answer": "1"}], role="fine-tuned",
            completed_job=job, expected_job_id=expected_job_id, artifact_version=artifact_version,
        )
    project.get_openai_client.assert_not_called()
    create.assert_not_called()


def test_fine_tuned_evaluation_verifies_artifact_version_before_chat_client_acquisition():
    namespace = inference_namespace()
    job = SimpleNamespace(
        id="offline-job", status="succeeded", model=namespace["MODEL"],
        fine_tuned_model="offline-trained-artifact",
        model_version="base-version-is-not-the-artifact-version",
    )
    deployment = SimpleNamespace(
        name="deployment", model_name=job.fine_tuned_model, model_version="artifact-version",
    )
    project, client, create = inference_project(deployment)
    events = []
    actual_validate = namespace["validate_math_deployment_identity"]

    def validate(*args, **kwargs):
        result = actual_validate(*args, **kwargs)
        events.append("verified")
        return result

    namespace["validate_math_deployment_identity"] = validate
    actual_client = project.get_openai_client.return_value

    def acquire_client():
        assert events == ["verified"]
        events.append("client")
        return actual_client

    project.get_openai_client.side_effect = acquire_client
    result = namespace["evaluate_math_deployment"](
        project, "deployment", [{"messages": [], "answer": "1"}], role="fine-tuned",
        completed_job=job, expected_job_id="offline-job", artifact_version="artifact-version",
    )
    assert result["verified_identity"]["model"] == job.fine_tuned_model
    assert result["verified_identity"]["version"] == "artifact-version"
    assert events == ["verified", "client", "verified"]
    create.assert_called_once()
    client.with_options.assert_called_once_with(max_retries=0)


def test_low_level_solver_cannot_bypass_deployment_identity_guard():
    namespace = inference_namespace()
    deployment = SimpleNamespace(name="deployment", model_name="unrelated-model", model_version="1")
    _project, client, create = inference_project(deployment)
    with pytest.raises(RuntimeError, match="identity is missing or mismatched"):
        namespace["solve_math"](
            client, deployment, {"messages": [], "answer": "1"},
            deployment_name="deployment", role="baseline",
        )
    client.with_options.assert_not_called()
    create.assert_not_called()


def test_notebook_entrypoint_uses_durable_state_and_remains_unexecuted():
    cells = notebook_cells()
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse(cell_source(cell))
            assert not cell.get("outputs")
            assert cell.get("execution_count") is None
    source = "\n".join(cell_source(cell) for cell in cells)
    assert 'STATE_PATH = DEMO_ROOT / "outputs/submission-state.json"' in source
    assert "fine_tuning_job = submit_or_resume_math_job(" in source
    assert "state_path=STATE_PATH" in source
    assert "project_endpoint=values[\"FOUNDRY_PROJECT_ENDPOINT\"]" in source
    assert 'assert len(training_rows) == 100' in source
    assert 'assert len(validation_rows) == 10' in source
    assert 'prepared_data = prepare_math_data(DEMO_ROOT, DEMO_ROOT / "outputs/prepared-data")' in source
    assert 'TRAIN_PATH = prepared_data["training_path"]' in source
    assert 'VALIDATION_PATH = prepared_data["validation_path"]' in source
    calls = [node for cell in cells if cell["cell_type"] == "code"
             for node in ast.walk(ast.parse(cell_source(cell))) if isinstance(node, ast.Call)]
    evaluations = [node for node in calls if isinstance(node.func, ast.Name)
                   and node.func.id == "evaluate_math_deployment"]
    assert len(evaluations) == 2
    by_role = {ast.literal_eval(next(item.value for item in node.keywords if item.arg == "role")): node
               for node in evaluations}
    assert set(by_role) == {"baseline", "fine-tuned"}
    keywords = {item.arg: item.value for item in by_role["fine-tuned"].keywords}
    assert isinstance(keywords["completed_job"], ast.Name) and keywords["completed_job"].id == "completed_job"
    assert ast.unparse(keywords["expected_job_id"]) == "fine_tuning_job.id"
    assert "FOUNDRY_MATH_FINE_TUNED_MODEL_VERSION" in ast.unparse(keywords["artifact_version"])


def test_every_chat_post_is_nested_in_the_identity_guarded_solver():
    functions = [node for cell in notebook_cells() if cell["cell_type"] == "code"
                 for node in ast.parse(cell_source(cell)).body if isinstance(node, ast.FunctionDef)]
    solvers = [node for node in functions if node.name == "solve_math"]
    assert len(solvers) == 1
    solver = solvers[0]
    guard_index = next(index for index, node in enumerate(solver.body) if any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id == "validate_math_deployment_identity" for call in ast.walk(node)
    ))
    post_index = next(index for index, node in enumerate(solver.body) if any(
        isinstance(call, ast.Call) and ast.unparse(call.func).endswith(".chat.completions.create")
        for call in ast.walk(node)
    ))
    assert guard_index < post_index
    posts = [node for cell in notebook_cells() if cell["cell_type"] == "code"
             for node in ast.walk(ast.parse(cell_source(cell)))
             if isinstance(node, ast.Call)
             and ast.unparse(node.func).endswith(".chat.completions.create")]
    assert len(posts) == 1