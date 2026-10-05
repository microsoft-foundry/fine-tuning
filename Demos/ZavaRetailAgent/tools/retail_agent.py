"""Interactive Zava retail agent using Microsoft Foundry SDK 2.x."""

import argparse
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


def _approve_mcp(model_client, response):
    approvals = [
        {
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": item.id,
        }
        for item in response.output
        if item.type == "mcp_approval_request"
    ]
    if approvals:
        return model_client.responses.create(
            input=approvals,
            previous_response_id=response.id,
        )
    return response


def main(model_name=None):
    policy = (ROOT / "data" / "policy.md").read_text(encoding="utf-8")
    model_name = model_name or required_env("FOUNDRY_MODEL_NAME")
    mcp_server_url = required_env("MCP_SERVER_URL")
    agent_name = f"{os.getenv('FOUNDRY_AGENT_NAME', 'zava-retail-agent')}-{uuid.uuid4().hex[:8]}"

    project_client = create_project_client()
    created_version = None
    original_endpoint = None
    try:
        created_version = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_name,
                instructions=policy,
                tools=[
                    MCPTool(
                        server_label="zava-retail",
                        server_url=mcp_server_url,
                        require_approval="always",
                    )
                ],
            ),
        )
        original_endpoint = project_client.agents.get(agent_name=agent_name).agent_endpoint
        project_client.agents.update_details(
            agent_name=agent_name,
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

        with create_openai_client(project_client, agent_name=agent_name) as model_client:
            previous_response_id = None
            print(f"{Fore.GREEN}Zava retail agent is ready. Type 'quit' to exit.{Style.RESET_ALL}")
            while True:
                user_input = input("YOU> ").strip()
                if user_input.lower() in {"quit", "exit"}:
                    break
                if not user_input:
                    continue
                response = model_client.responses.create(
                    input=user_input,
                    previous_response_id=previous_response_id,
                )
                response = _approve_mcp(model_client, response)
                previous_response_id = response.id
                print(f"ZAVA> {response.output_text}")
    finally:
        if original_endpoint is not None:
            project_client.agents.update_details(
                agent_name=agent_name,
                agent_endpoint=original_endpoint,
            )
        if created_version is not None:
            project_client.agents.delete_version(
                agent_name=agent_name,
                agent_version=created_version.version,
                force=True,
            )
        project_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()
    main(model_name=args.model)
