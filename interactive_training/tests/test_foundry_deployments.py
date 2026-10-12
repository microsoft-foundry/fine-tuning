"""Offline Fireworks deployment contract tests."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from interactive_training.deployment import DeploymentTarget, create_deployment, delete_deployment
from interactive_training.deployment import foundry_deployments as deployment


PROJECT_ID = (
    "/subscriptions/sub/resourceGroups/group/providers/"
    "Microsoft.CognitiveServices/accounts/account/projects/project"
)
ACCOUNT_ID = PROJECT_ID.rsplit("/projects/", 1)[0]
ENDPOINT = "https://account.services.ai.azure.com/api/projects/project"
MODEL = "model:ckpt-step-final"
SUCCEEDED = {
    "name": "model",
    "sku": {"capacity": 40},
    "properties": {"model": {"name": MODEL}, "provisioningState": "Succeeded"},
}


def checkpoint(name, created):
    return SimpleNamespace(fine_tuned_model_checkpoint=name, created_at=created)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.openai = self.client.__enter__.return_value.get_openai_client.return_value.__enter__.return_value
        self.openai.fine_tuning.jobs.checkpoints.list.return_value = [
            checkpoint(MODEL, 1),
            checkpoint("model:ckpt-step-99", 99),
        ]
        self.credential = patch.object(deployment, "DefaultAzureCredential").start()
        self.factory = patch.object(deployment, "AIProjectClient", return_value=self.client).start()
        self.arm_patcher = patch.object(deployment, "_arm_request")
        self.arm = self.arm_patcher.start()
        self.addCleanup(patch.stopall)

    def test_creates_final_checkpoint_and_returns_arm_deployment(self):
        creating = {"properties": {"provisioningState": "Creating"}}
        self.arm.side_effect = [{}, creating, SUCCEEDED]
        with patch.object(deployment.time, "sleep"):
            result = create_deployment(PROJECT_ID, "session_123", DeploymentTarget.FIREWORKS, 40)
        self.assertIs(result, SUCCEEDED)
        self.assertEqual([call.args[1] for call in self.arm.call_args_list], ["GET", "PUT", "GET"])
        body = self.arm.call_args_list[1].args[3]
        self.assertEqual(body["properties"]["model"]["name"], MODEL)
        self.assertEqual(body["properties"]["model"]["sourceAccount"], ACCOUNT_ID)
        self.assertEqual(body["sku"], {"name": "GlobalProvisionedManaged", "capacity": 40})
        self.factory.assert_called_once_with(
            endpoint=ENDPOINT, credential=self.credential.return_value.__enter__.return_value
        )

    def test_falls_back_to_newest_or_last_when_timestamps_missing(self):
        self.openai.fine_tuning.jobs.checkpoints.list.return_value = [
            checkpoint("model:ckpt-step-99", None),
            checkpoint("model:ckpt-step-2", 10),
            checkpoint("model:ckpt-step-3", 10),
        ]
        self.arm.side_effect = [{}, SUCCEEDED]
        create_deployment(PROJECT_ID, "model_123", DeploymentTarget.FIREWORKS, 40)
        self.assertEqual(self.arm.call_args_list[1].args[3]["properties"]["model"]["name"], "model:ckpt-step-3")

    def test_uses_last_final_if_all_dates_missing(self):
        self.openai.fine_tuning.jobs.checkpoints.list.return_value = [
            checkpoint("model:ckpt-final-1", None), checkpoint("model:ckpt-final-2", None)
        ]
        self.arm.side_effect = [{}, SUCCEEDED]
        create_deployment(PROJECT_ID, "session_123", DeploymentTarget.FIREWORKS, 40)
        self.assertEqual(self.arm.call_args_list[1].args[3]["properties"]["model"]["name"], "model:ckpt-final-2")

    def test_no_checkpoint_does_not_create_deployment(self):
        self.openai.fine_tuning.jobs.checkpoints.list.return_value = []
        with self.assertRaisesRegex(ValueError, "No checkpoints"):
            create_deployment(PROJECT_ID, "session_123", DeploymentTarget.FIREWORKS, 40)
        self.arm.assert_not_called()

    def test_existing_matching_deployment_is_reused(self):
        self.arm.return_value = SUCCEEDED
        self.assertIs(create_deployment(PROJECT_ID, "session_123", DeploymentTarget.FIREWORKS, 40), SUCCEEDED)
        self.arm.assert_called_once()

    def test_existing_deployment_not_overwritten(self):
        for other in (
            {**SUCCEEDED, "properties": {"model": {"name": "different"}}},
            {**SUCCEEDED, "sku": {"capacity": 50}},
        ):
            with self.subTest(other=other):
                self.arm.reset_mock(return_value=True)
                self.arm.return_value = other
                with self.assertRaisesRegex(RuntimeError, "already"):
                    create_deployment(PROJECT_ID, "session_123", DeploymentTarget.FIREWORKS, 40)
                self.arm.assert_called_once()

    def test_failed_deployment_is_not_deleted(self):
        self.arm.return_value = {
            **SUCCEEDED, "properties": {"model": {"name": MODEL}, "provisioningState": "Failed"}
        }
        with self.assertRaisesRegex(RuntimeError, "Failed"):
            create_deployment(PROJECT_ID, "session_123", DeploymentTarget.FIREWORKS, 40)
        self.arm.assert_called_once()

    def test_api_key_environment_does_not_override_default_credential(self):
        self.arm.side_effect = [{}, SUCCEEDED]
        with patch.dict(os.environ, {"AZURE_AI_API_KEY": "test-key"}):
            create_deployment(PROJECT_ID, "session_123", DeploymentTarget.FIREWORKS, 40)
        self.factory.assert_called_once_with(
            endpoint=ENDPOINT, credential=self.credential.return_value.__enter__.return_value
        )
        self.client.__enter__.return_value.get_openai_client.assert_called_once_with()
        self.assertEqual(
            self.arm.call_args_list[1].args[0], self.credential.return_value.__enter__.return_value
        )

    def test_invalid_inputs_do_not_call_cloud(self):
        for project, session, target, capacity in (
            ("/subscriptions/invalid", "session_123", DeploymentTarget.FIREWORKS, 40),
            (PROJECT_ID, "bad", DeploymentTarget.FIREWORKS, 40),
            (PROJECT_ID, "session_123", "invalid", 40),
            (PROJECT_ID, "session_123", DeploymentTarget.FIREWORKS, 0),
            (PROJECT_ID, "session_123", DeploymentTarget.FIREWORKS, True),
        ):
            with self.subTest(project=project, session=session, target=target, capacity=capacity):
                with self.assertRaises(ValueError):
                    create_deployment(project, session, target, capacity)
        self.factory.assert_not_called()

    def test_project_id_derives_account_and_endpoint(self):
        self.assertEqual(deployment._project_parts(PROJECT_ID), (ACCOUNT_ID, ENDPOINT))
        with self.assertRaises(ValueError):
            deployment._project_parts(PROJECT_ID + "/extra")
        with self.assertRaises(ValueError):
            deployment._deployment_url(ACCOUNT_ID, "../other")

    def test_delete_waits_until_not_found(self):
        self.arm.side_effect = [SUCCEEDED, {}, SUCCEEDED, {}]
        with patch.object(deployment.time, "sleep") as sleep:
            delete_deployment(PROJECT_ID, "model")
        self.assertEqual([call.args[1] for call in self.arm.call_args_list], ["GET", "DELETE", "GET", "GET"])
        sleep.assert_called_once()
        self.assertIn(f"{ACCOUNT_ID}/deployments/model?", self.arm.call_args_list[1].args[2])

    def test_delete_rejects_missing_or_invalid_name(self):
        self.arm.return_value = {}
        with self.assertRaisesRegex(ValueError, "does not exist"):
            delete_deployment(PROJECT_ID, "missing")
        self.arm.assert_called_once()
        with self.assertRaisesRegex(ValueError, "deployment name"):
            delete_deployment(PROJECT_ID, "../other")

    def test_arm_delete_accepts_empty_body_and_get_404(self):
        self.arm_patcher.stop()
        credential = MagicMock()
        credential.get_token.return_value.token = "local-test-token"
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b""
        with patch.object(deployment, "urlopen", return_value=response) as send:
            self.assertEqual(deployment._arm_request(credential, "DELETE", "https://management.azure.com/test"), {})
        self.assertEqual(send.call_args.args[0].get_header("Authorization"), "Bearer local-test-token")
        with patch.object(deployment, "urlopen", side_effect=HTTPError(
            "https://management.azure.com/test", 404, "Not Found", {}, None
        )):
            self.assertEqual(deployment._arm_request(credential, "GET", "https://management.azure.com/test"), {})


if __name__ == "__main__":
    unittest.main()
