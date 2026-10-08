from types import SimpleNamespace

import pytest
from azure.core.exceptions import ResourceNotFoundError

from shared.deployment import ensure_deployment


def test_deployment_creation_requires_not_found() -> None:
    created = []

    class ForbiddenError(RuntimeError):
        status_code = 403

    def get(name):
        raise ForbiddenError("Access denied")

    project = SimpleNamespace(deployments=SimpleNamespace(get=get))
    with pytest.raises(ForbiddenError):
        ensure_deployment(
            project, deployment_name="customer-model", create=lambda: created.append(True)
        )
    assert not created


def test_not_found_can_use_explicit_creation_callback() -> None:
    created = []
    deployment = SimpleNamespace(model_name="fine-tuned-model")

    def get(name):
        if not created:
            raise ResourceNotFoundError("Not found")
        return deployment

    project = SimpleNamespace(deployments=SimpleNamespace(get=get))
    result = ensure_deployment(
        project,
        deployment_name="customer-model",
        expected_model_name="fine-tuned-model",
        create=lambda: created.append(True),
    )
    assert result is deployment
    assert created == [True]
