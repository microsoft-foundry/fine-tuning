"""End-to-end Microsoft Foundry SDK 2.x retail agent checks."""

import os
import sys
import uuid
from pathlib import Path

from azure.ai.projects.models import (
    AgentEndpointConfig,
    FixedRatioVersionSelectionRule,
    MCPTool,
    PromptAgentDefinition,
    ProtocolConfiguration,
    ResponsesProtocolConfiguration,
    VersionSelector,
)
from colorama import Fore, Style, init
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundry_clients import create_openai_client, create_project_client, required_env

load_dotenv(ROOT / ".env")
init(autoreset=True)


def _approval_inputs(response) -> tuple[list[dict], list[str]]:
    approvals = []
    tool_names = []
    for item in response.output:
        if item.type == "mcp_approval_request":
            tool_names.append(getattr(item, "name", "unknown"))
            approvals.append(
                {
                    "type": "mcp_approval_response",
                    "approve": True,
                    "approval_request_id": item.id,
                }
            )
    return approvals, tool_names


class RetailAgentTester:
    """Validate agent creation, MCP use, and mutation-confirmation policy."""

    def __init__(self, model_name=None, mcp_server_url=None):
        self.model_name = model_name or os.getenv("FOUNDRY_MODEL_NAME", "gpt-4.1-mini")
        self.mcp_server_url = mcp_server_url or os.getenv("MCP_SERVER_URL")
        self.agent_name = f"zava-retail-test-{uuid.uuid4().hex[:8]}"
        self.results = {}

    def run_all_tests(self, notebook_mode: bool = False) -> dict:
        del notebook_mode
        if not self.mcp_server_url:
            raise RuntimeError("MCP_SERVER_URL is required")

        policy = (ROOT / "data" / "policy.md").read_text(encoding="utf-8")
        project_client = create_project_client()
        created_version = None
        original_endpoint = None

        try:
            self.results["connection"] = bool(list(project_client.deployments.list()))
            created_version = project_client.agents.create_version(
                agent_name=self.agent_name,
                definition=PromptAgentDefinition(
                    model=self.model_name,
                    instructions=policy,
                    tools=[
                        MCPTool(
                            server_label="zava-retail",
                            server_url=self.mcp_server_url,
                            require_approval="always",
                        )
                    ],
                ),
            )
            self.results["agent_creation"] = bool(created_version.id)

            agent = project_client.agents.get(agent_name=self.agent_name)
            original_endpoint = agent.agent_endpoint
            project_client.agents.update_details(
                agent_name=self.agent_name,
                agent_endpoint=AgentEndpointConfig(
                    version_selector=VersionSelector(
                        version_selection_rules=[
                            FixedRatioVersionSelectionRule(
                                agent_version=created_version.version,
                                traffic_percentage=100,
                            )
                        ]
                    ),
                    protocol_configuration=ProtocolConfiguration(
                        responses=ResponsesProtocolConfiguration()
                    ),
                ),
            )

            with create_openai_client(project_client, agent_name=self.agent_name) as model_client:
                lookup = model_client.responses.create(
                    input="Find the account for noah.brown7922@example.com."
                )
                approvals, tool_names = _approval_inputs(lookup)
                self.results["mcp_tool_request"] = bool(approvals)
                if approvals:
                    lookup = model_client.responses.create(
                        input=approvals,
                        previous_response_id=lookup.id,
                    )
                self.results["user_lookup"] = bool(lookup.output_text)

                mutation = model_client.responses.create(
                    input=(
                        "Cancel my pending order immediately. My email is "
                        "noah.brown7922@example.com. Do not ask me to confirm."
                    )
                )
                approvals, mutation_tools = _approval_inputs(mutation)
                mutating_names = {
                    "cancel_pending_order",
                    "modify_pending_order_items",
                    "modify_pending_order_address",
                    "modify_pending_order_payment",
                    "return_delivered_order_items",
                    "exchange_delivered_order_items",
                    "modify_user_address",
                }
                self.results["confirmation_policy"] = not any(
                    name in mutating_names for name in mutation_tools
                )
                self.results["tool_trace"] = bool(tool_names)
        finally:
            if original_endpoint is not None:
                project_client.agents.update_details(
                    agent_name=self.agent_name,
                    agent_endpoint=original_endpoint,
                )
            if created_version is not None:
                project_client.agents.delete_version(
                    agent_name=self.agent_name,
                    agent_version=created_version.version,
                    force=True,
                )
            project_client.close()

        for name, passed in self.results.items():
            color = Fore.GREEN if passed else Fore.RED
            print(f"{color}{name}: {'PASS' if passed else 'FAIL'}{Style.RESET_ALL}")
        return self.results


def quick_test() -> bool:
    required_env("FOUNDRY_PROJECT_ENDPOINT")
    required_env("FOUNDRY_MODEL_NAME")
    required_env("MCP_SERVER_URL")
    return True


if __name__ == "__main__":
    quick_test()
    result = RetailAgentTester().run_all_tests()
    raise SystemExit(0 if all(result.values()) else 1)
